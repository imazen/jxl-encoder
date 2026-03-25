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

use crate::api::{EncodeError, LosslessConfig, LossyConfig, PixelLayout};

/// Encode RGB8 pixels to lossy JXL.
pub fn encode_rgb8(img: ImgRef<Rgb<u8>>, config: &LossyConfig) -> Result<Vec<u8>, EncodeError> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes: &[u8] = bytemuck::cast_slice(&buf);
    config
        .encode(bytes, w as u32, h as u32, PixelLayout::Rgb8)
        .map_err(|e| e.decompose().0)
}

/// Encode RGBA8 pixels to lossy JXL.
pub fn encode_rgba8(img: ImgRef<Rgba<u8>>, config: &LossyConfig) -> Result<Vec<u8>, EncodeError> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes: &[u8] = bytemuck::cast_slice(&buf);
    config
        .encode(bytes, w as u32, h as u32, PixelLayout::Rgba8)
        .map_err(|e| e.decompose().0)
}

/// Encode Gray8 pixels to lossy JXL (expanded to RGB).
pub fn encode_gray8(img: ImgRef<Gray<u8>>, config: &LossyConfig) -> Result<Vec<u8>, EncodeError> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes = gray_to_rgb_bytes(&buf);
    config
        .encode(&bytes, w as u32, h as u32, PixelLayout::Rgb8)
        .map_err(|e| e.decompose().0)
}

/// Encode BGRA8 pixels to lossy JXL (native BGRA path, no swizzle).
pub fn encode_bgra8(img: ImgRef<BGRA<u8>>, config: &LossyConfig) -> Result<Vec<u8>, EncodeError> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes: &[u8] = bytemuck::cast_slice(&buf);
    config
        .encode(bytes, w as u32, h as u32, PixelLayout::Bgra8)
        .map_err(|e| e.decompose().0)
}

/// Encode RGB8 pixels to lossless JXL.
pub fn encode_rgb8_lossless(
    img: ImgRef<Rgb<u8>>,
    config: &LosslessConfig,
) -> Result<Vec<u8>, EncodeError> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes: &[u8] = bytemuck::cast_slice(&buf);
    config
        .encode(bytes, w as u32, h as u32, PixelLayout::Rgb8)
        .map_err(|e| e.decompose().0)
}

/// Encode RGBA8 pixels to lossless JXL.
pub fn encode_rgba8_lossless(
    img: ImgRef<Rgba<u8>>,
    config: &LosslessConfig,
) -> Result<Vec<u8>, EncodeError> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes: &[u8] = bytemuck::cast_slice(&buf);
    config
        .encode(bytes, w as u32, h as u32, PixelLayout::Rgba8)
        .map_err(|e| e.decompose().0)
}

/// Encode BGRA8 pixels to lossless JXL (native BGRA path, no swizzle).
pub fn encode_bgra8_lossless(
    img: ImgRef<BGRA<u8>>,
    config: &LosslessConfig,
) -> Result<Vec<u8>, EncodeError> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes: &[u8] = bytemuck::cast_slice(&buf);
    config
        .encode(bytes, w as u32, h as u32, PixelLayout::Bgra8)
        .map_err(|e| e.decompose().0)
}

/// Encode Gray8 pixels to lossless JXL.
pub fn encode_gray8_lossless(
    img: ImgRef<Gray<u8>>,
    config: &LosslessConfig,
) -> Result<Vec<u8>, EncodeError> {
    let (buf, w, h) = img.to_contiguous_buf();
    let bytes: &[u8] = bytemuck::cast_slice(&buf);
    config
        .encode(bytes, w as u32, h as u32, PixelLayout::Gray8)
        .map_err(|e| e.decompose().0)
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
