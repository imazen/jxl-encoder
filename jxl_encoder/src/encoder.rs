// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! High-level JPEG XL encoder API.
//!
//! This module provides a simple interface for encoding images to JXL format.

use crate::bit_writer::BitWriter;
use crate::error::Result;
use crate::frame::{FrameEncoder, FrameEncoderOptions};
use crate::headers::{ColorEncoding, FileHeader};
use crate::modular::ModularImage;

/// sRGB to linear conversion (exact IEC 61966-2-1 transfer function).
#[inline]
fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// JPEG XL encoder options.
///
/// Prefer [`LosslessConfig`](crate::LosslessConfig) or [`LossyConfig`](crate::LossyConfig).
#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct EncoderOptions {
    /// Distance parameter (0.0 = lossless, higher = more lossy).
    pub distance: f32,
    /// Effort level (1-10, higher = better compression, slower).
    pub effort: u8,
    /// Use modular mode for all encoding.
    pub force_modular: bool,
    /// Use ANS entropy coding instead of Huffman (4-10% smaller, default true).
    pub use_ans: bool,
    /// Optimize Huffman/ANS codes (two-pass, default true).
    pub optimize_codes: bool,
    /// Use custom coefficient ordering (default true).
    pub custom_orders: bool,
    /// Enable noise synthesis (estimates and encodes noise params).
    pub enable_noise: bool,
    /// Enable Wiener denoising pre-filter (implies enable_noise).
    pub enable_denoise: bool,
    /// Enable gaborish inverse pre-filter (default true).
    pub enable_gaborish: bool,
    /// Enable content-adaptive MA tree learning for modular encoding.
    /// Off by default (uses fixed gradient tree).
    pub use_tree_learning: bool,
    /// Enable squeeze (Haar wavelet) transform for modular encoding.
    /// Off by default. Enables progressive decoding.
    /// Enable squeeze (Haar wavelet) transform for modular encoding.
    /// Off by default. Enables progressive decoding.
    pub use_squeeze: bool,
}

impl Default for EncoderOptions {
    fn default() -> Self {
        Self {
            distance: 0.0, // Lossless by default
            effort: 7,
            force_modular: false,
            use_ans: true,
            optimize_codes: true,
            custom_orders: true,
            enable_noise: false,
            enable_denoise: false,
            enable_gaborish: true,
            use_tree_learning: false,
            use_squeeze: false,
        }
    }
}

impl EncoderOptions {
    /// Creates options for lossless encoding.
    pub fn lossless() -> Self {
        Self {
            distance: 0.0,
            ..Default::default()
        }
    }

    /// Creates options for lossy encoding with given distance.
    pub fn lossy(distance: f32) -> Self {
        Self {
            distance,
            ..Default::default()
        }
    }
}

/// JPEG XL encoder.
///
/// Prefer [`LosslessConfig`](crate::LosslessConfig) or [`LossyConfig`](crate::LossyConfig).
#[doc(hidden)]
pub struct Encoder {
    options: EncoderOptions,
}

impl Encoder {
    /// Creates a new encoder with default options.
    pub fn new() -> Self {
        Self::with_options(EncoderOptions::default())
    }

    /// Creates a new encoder with custom options.
    pub fn with_options(options: EncoderOptions) -> Self {
        Self { options }
    }

    /// Encodes an RGB8 image to JXL format.
    pub fn encode_rgb8(&self, data: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        let image = ModularImage::from_rgb8(data, width, height)?;
        self.encode_modular_image(&image)
    }

    /// Encodes an RGBA8 image to JXL format.
    pub fn encode_rgba8(&self, data: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        let image = ModularImage::from_rgba8(data, width, height)?;
        self.encode_modular_image(&image)
    }

    /// Encodes a grayscale 8-bit image to JXL format.
    pub fn encode_gray8(&self, data: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        let image = ModularImage::from_gray8(data, width, height)?;
        self.encode_modular_image(&image)
    }

    /// Encodes a 16-bit RGB image to JXL format (lossless).
    pub fn encode_rgb16_native(&self, data: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        let image = ModularImage::from_rgb16_native(data, width, height)?;
        self.encode_modular_image(&image)
    }

    /// Encodes a 16-bit RGBA image to JXL format (lossless).
    pub fn encode_rgba16_native(
        &self,
        data: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>> {
        let image = ModularImage::from_rgba16_native(data, width, height)?;
        self.encode_modular_image(&image)
    }

    /// Encodes a 16-bit grayscale image to JXL format (lossless).
    pub fn encode_gray16_native(
        &self,
        data: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>> {
        let image = ModularImage::from_gray16_native(data, width, height)?;
        self.encode_modular_image(&image)
    }

    /// Encodes an RGB8 image to JXL format with lossy compression.
    #[doc(hidden)]
    ///
    /// # Arguments
    /// * `data` - RGB8 pixel data in row-major order (R,G,B,R,G,B,...)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `distance` - Butteraugli distance (0.0 = lossless, 1.0 = high quality, higher = smaller files)
    pub fn encode_lossy_rgb8(
        &self,
        data: &[u8],
        width: usize,
        height: usize,
        distance: f32,
    ) -> Result<Vec<u8>> {
        use crate::tiny::TinyEncoder;

        // Convert sRGB u8 to linear f32
        let linear_rgb: Vec<f32> = data
            .chunks(3)
            .flat_map(|px| {
                [
                    srgb_to_linear(px[0]),
                    srgb_to_linear(px[1]),
                    srgb_to_linear(px[2]),
                ]
            })
            .collect();

        // Configure TinyEncoder with options
        let mut encoder = TinyEncoder::new(distance);
        encoder.use_ans = self.options.use_ans;
        encoder.optimize_codes = self.options.optimize_codes;
        encoder.custom_orders = self.options.custom_orders;
        encoder.enable_noise = self.options.enable_noise || self.options.enable_denoise;
        encoder.enable_denoise = self.options.enable_denoise;
        encoder.enable_gaborish = self.options.enable_gaborish;

        encoder.encode(width, height, &linear_rgb, None)
    }

    /// Encodes a ModularImage to JXL format.
    pub fn encode_modular_image(&self, image: &ModularImage) -> Result<Vec<u8>> {
        let mut writer = BitWriter::new();

        // Write file header (includes JXL signature)
        let mut file_header = if image.is_grayscale {
            FileHeader::new_gray(image.width() as u32, image.height() as u32)
        } else if image.has_alpha {
            FileHeader::new_rgba(image.width() as u32, image.height() as u32)
        } else {
            FileHeader::new_rgb(image.width() as u32, image.height() as u32)
        };

        // Set bit depth from image
        if image.bit_depth == 16 {
            file_header.metadata.bit_depth = crate::headers::file_header::BitDepth::uint16();
        }

        file_header.write(&mut writer)?;

        // JXL frame header has #[aligned] attribute, meaning it starts byte-aligned
        writer.zero_pad_to_byte();

        // Encode frame
        let frame_options = FrameEncoderOptions {
            use_modular: self.options.distance == 0.0 || self.options.force_modular,
            effort: self.options.effort,
            use_ans: self.options.use_ans,
            use_tree_learning: self.options.use_tree_learning,
            use_squeeze: self.options.use_squeeze,
        };

        let frame_encoder = FrameEncoder::new(image.width(), image.height(), frame_options);

        let color_encoding = ColorEncoding::srgb();
        frame_encoder.encode_modular(image, &color_encoding, &mut writer)?;

        // Finish with byte padding
        let bytes = writer.finish_with_padding();
        Ok(bytes)
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Encodes RGB8 data to JXL with default options.
///
/// This is a convenience function for the common case of encoding
/// an 8-bit RGB image to lossless JXL.
#[doc(hidden)]
pub fn encode_rgb8(data: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    Encoder::new().encode_rgb8(data, width, height)
}

/// Encodes RGBA8 data to JXL with default options.
#[doc(hidden)]
pub fn encode_rgba8(data: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    Encoder::new().encode_rgba8(data, width, height)
}

/// Encodes RGB8 data to JXL with lossy compression.
#[doc(hidden)]
pub fn encode_lossy_rgb8(
    data: &[u8],
    width: usize,
    height: usize,
    distance: f32,
) -> Result<Vec<u8>> {
    Encoder::with_options(EncoderOptions::lossy(distance))
        .encode_lossy_rgb8(data, width, height, distance)
}

// Tests are in a separate file to keep this module focused on implementation
#[cfg(test)]
#[path = "encoder_tests.rs"]
mod encoder_tests;
