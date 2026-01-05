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

/// JPEG XL encoder options.
#[derive(Debug, Clone)]
pub struct EncoderOptions {
    /// Distance parameter (0.0 = lossless, higher = more lossy).
    pub distance: f32,
    /// Effort level (1-10, higher = better compression, slower).
    pub effort: u8,
    /// Use modular mode for all encoding.
    pub force_modular: bool,
}

impl Default for EncoderOptions {
    fn default() -> Self {
        Self {
            distance: 0.0, // Lossless by default
            effort: 7,
            force_modular: false,
        }
    }
}

impl EncoderOptions {
    /// Creates options for lossless encoding.
    pub fn lossless() -> Self {
        Self {
            distance: 0.0,
            effort: 7,
            force_modular: false,
        }
    }

    /// Creates options for lossy encoding with given distance.
    pub fn lossy(distance: f32) -> Self {
        Self {
            distance,
            effort: 7,
            force_modular: false,
        }
    }
}

/// JPEG XL encoder.
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

    /// Encodes an RGB8 image to JXL format with lossy compression.
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
        use crate::color::xyb::srgb_image_to_xyb;

        // Convert u8 to f32
        let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

        // Separate RGB channels
        let num_pixels = width * height;
        let mut r = vec![0.0f32; num_pixels];
        let mut g = vec![0.0f32; num_pixels];
        let mut b = vec![0.0f32; num_pixels];
        for i in 0..num_pixels {
            r[i] = data_f32[i * 3];
            g[i] = data_f32[i * 3 + 1];
            b[i] = data_f32[i * 3 + 2];
        }

        // Convert to XYB color space
        let mut x_out = vec![0.0f32; num_pixels];
        let mut y_out = vec![0.0f32; num_pixels];
        let mut b_out = vec![0.0f32; num_pixels];
        srgb_image_to_xyb(&r, &g, &b, &mut x_out, &mut y_out, &mut b_out);

        // Interleave XYB channels back for VarDCT encoding
        let mut xyb_data = vec![0.0f32; num_pixels * 3];
        for i in 0..num_pixels {
            xyb_data[i * 3] = x_out[i];
            xyb_data[i * 3 + 1] = y_out[i];
            xyb_data[i * 3 + 2] = b_out[i];
        }

        let mut writer = BitWriter::new();

        // Write file header (for lossy, uses XYB color encoding)
        let file_header = FileHeader::new_rgb_lossy(width as u32, height as u32);
        file_header.write(&mut writer)?;

        // Byte align before frame
        writer.zero_pad_to_byte();

        // Encode VarDCT frame
        let frame_options = FrameEncoderOptions {
            use_modular: false,
            effort: self.options.effort,
        };
        let frame_encoder = FrameEncoder::new(width, height, frame_options);
        let color_encoding = ColorEncoding::srgb();
        frame_encoder.encode_vardct(&xyb_data, distance, &color_encoding, &mut writer)?;

        let bytes = writer.finish_with_padding();
        Ok(bytes)
    }

    /// Encodes a ModularImage to JXL format.
    pub fn encode_modular_image(&self, image: &ModularImage) -> Result<Vec<u8>> {
        let mut writer = BitWriter::new();

        // Write file header (includes JXL signature)
        let file_header = if image.is_grayscale {
            FileHeader::new_gray(image.width() as u32, image.height() as u32)
        } else if image.has_alpha {
            FileHeader::new_rgba(image.width() as u32, image.height() as u32)
        } else {
            FileHeader::new_rgb(image.width() as u32, image.height() as u32)
        };
        file_header.write(&mut writer)?;

        // JXL frame header has #[aligned] attribute, meaning it starts byte-aligned
        writer.zero_pad_to_byte();

        // Encode frame
        let frame_options = FrameEncoderOptions {
            use_modular: self.options.distance == 0.0 || self.options.force_modular,
            effort: self.options.effort,
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
pub fn encode_rgb8(data: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    Encoder::new().encode_rgb8(data, width, height)
}

/// Encodes RGBA8 data to JXL with default options.
pub fn encode_rgba8(data: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    Encoder::new().encode_rgba8(data, width, height)
}

/// Encodes RGB8 data to JXL with lossy compression.
///
/// # Arguments
/// * `data` - RGB8 pixel data in row-major order
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `distance` - Butteraugli distance (1.0 = high quality, higher = smaller files)
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
