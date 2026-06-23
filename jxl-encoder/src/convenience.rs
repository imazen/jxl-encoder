// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Convenience functions for encoding pixel buffers to JPEG XL.
//!
//! These functions wrap [`LossyConfig`] and [`LosslessConfig`] with typed
//! pixel-format helpers so callers can encode an [`ImgRef`] in a single call
//! without manually extracting bytes or choosing a [`PixelLayout`].
//!
//! Enable the `convenience` feature to use this module.

use alloc::vec::Vec;

use imgref::ImgRef;
use rgb::alt::BGRA;
use rgb::{Gray, Rgb, Rgba};

use crate::api::{At, EncodeError, LosslessConfig, LossyConfig, PixelLayout};

/// Encode RGB8 pixels to lossy JXL.
pub fn encode_rgb8(img: ImgRef<Rgb<u8>>, config: &LossyConfig) -> Result<Vec<u8>, At<EncodeError>> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes: &[u8] = bytemuck::cast_slice(&buf);
    config.encode(bytes, w as u32, h as u32, PixelLayout::Rgb8)
}

/// Encode RGBA8 pixels to lossy JXL.
pub fn encode_rgba8(
    img: ImgRef<Rgba<u8>>,
    config: &LossyConfig,
) -> Result<Vec<u8>, At<EncodeError>> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes: &[u8] = bytemuck::cast_slice(&buf);
    config.encode(bytes, w as u32, h as u32, PixelLayout::Rgba8)
}

/// Encode Gray8 pixels to lossy JXL (expanded to RGB).
pub fn encode_gray8(
    img: ImgRef<Gray<u8>>,
    config: &LossyConfig,
) -> Result<Vec<u8>, At<EncodeError>> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes = gray_to_rgb_bytes(&buf);
    config.encode(&bytes, w as u32, h as u32, PixelLayout::Rgb8)
}

/// Encode BGRA8 pixels to lossy JXL (native BGRA path, no swizzle).
pub fn encode_bgra8(
    img: ImgRef<BGRA<u8>>,
    config: &LossyConfig,
) -> Result<Vec<u8>, At<EncodeError>> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes: &[u8] = bytemuck::cast_slice(&buf);
    config.encode(bytes, w as u32, h as u32, PixelLayout::Bgra8)
}

/// Encode RGB8 pixels to lossless JXL.
pub fn encode_rgb8_lossless(
    img: ImgRef<Rgb<u8>>,
    config: &LosslessConfig,
) -> Result<Vec<u8>, At<EncodeError>> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes: &[u8] = bytemuck::cast_slice(&buf);
    config.encode(bytes, w as u32, h as u32, PixelLayout::Rgb8)
}

/// Encode RGBA8 pixels to lossless JXL.
pub fn encode_rgba8_lossless(
    img: ImgRef<Rgba<u8>>,
    config: &LosslessConfig,
) -> Result<Vec<u8>, At<EncodeError>> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes: &[u8] = bytemuck::cast_slice(&buf);
    config.encode(bytes, w as u32, h as u32, PixelLayout::Rgba8)
}

/// Encode BGRA8 pixels to lossless JXL (native BGRA path, no swizzle).
pub fn encode_bgra8_lossless(
    img: ImgRef<BGRA<u8>>,
    config: &LosslessConfig,
) -> Result<Vec<u8>, At<EncodeError>> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes: &[u8] = bytemuck::cast_slice(&buf);
    config.encode(bytes, w as u32, h as u32, PixelLayout::Bgra8)
}

/// Encode Gray8 pixels to lossless JXL.
pub fn encode_gray8_lossless(
    img: ImgRef<Gray<u8>>,
    config: &LosslessConfig,
) -> Result<Vec<u8>, At<EncodeError>> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes: &[u8] = bytemuck::cast_slice(&buf);
    config.encode(bytes, w as u32, h as u32, PixelLayout::Gray8)
}

/// Expand grayscale pixels to RGB bytes (3 bytes per pixel).
pub(crate) fn gray_to_rgb_bytes(pixels: &[Gray<u8>]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(pixels.len() * 3);
    for g in pixels {
        let v = g.value();
        bytes.push(v);
        bytes.push(v);
        bytes.push(v);
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard for the convenience wrappers' trace preservation.
    /// Each `encode_*` now returns `*Config::encode(...)`'s `At<EncodeError>`
    /// DIRECTLY (no `.map_err(|e| e.decompose().0)`), so any encode error must
    /// carry the whereat trace. We exercise that delegated method with a buffer
    /// too small for the claimed dimensions (`InvalidInput`) and assert the
    /// trace survives. (A convenience fn can't be driven to error through
    /// `ImgRef`, which rejects degenerate dimensions at construction.)
    #[test]
    fn convenience_encode_error_preserves_whereat_trace() {
        let cfg = LossyConfig::new(1.0);
        // 3 bytes cannot hold a 4×4 RGB image (needs 48) → InvalidInput.
        let err = cfg.encode(&[0u8; 3], 4, 4, PixelLayout::Rgb8).unwrap_err();
        assert!(
            err.frame_count() >= 1,
            "whereat trace lost on encode error: {err:?}"
        );
    }
}
