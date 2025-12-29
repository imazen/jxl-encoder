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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_small_rgb() {
        // 2x2 RGB image
        let data = vec![
            255, 0, 0, // Red
            0, 255, 0, // Green
            0, 0, 255, // Blue
            255, 255, 0, // Yellow
        ];

        let encoded = encode_rgb8(&data, 2, 2).unwrap();

        // Check JXL signature
        assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);

        // Should have produced some output
        assert!(encoded.len() > 10);

        // Debug output
        eprintln!("Encoded {} bytes:", encoded.len());
        for (i, b) in encoded.iter().enumerate() {
            eprint!("{:02x} ", b);
            if (i + 1) % 16 == 0 {
                eprintln!();
            }
        }
        eprintln!();

        // Write to /tmp for debugging
        std::fs::write("/tmp/test_out.jxl", &encoded).unwrap();
    }

    #[test]
    fn test_encode_pattern() {
        // 4x4 pattern with 4 unique values (max for simple Huffman)
        let mut data = Vec::with_capacity(4 * 4 * 3);
        for y in 0..4 {
            for x in 0..4 {
                // Use only 4 unique values: 0, 64, 128, 192
                let v = ((x % 2) * 64 + (y % 2) * 128) as u8;
                data.push(v);
                data.push(v);
                data.push(v);
            }
        }

        let encoded = encode_rgb8(&data, 4, 4).unwrap();
        // Check JXL signature
        assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);
    }

    #[test]
    fn test_encode_flat() {
        // 4x4 flat gray image
        let data = vec![128u8; 4 * 4 * 3];

        let encoded = encode_rgb8(&data, 4, 4).unwrap();
        // Check JXL signature
        assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);

        // Flat images should compress very well
        // (though our minimal encoder may not be optimal)
    }

    #[test]
    fn test_encode_black_2x2() {
        // 2x2 all-black image - should work with minimal zero-everywhere encoding
        let data = vec![0u8; 2 * 2 * 3]; // All zeros

        let encoded = encode_rgb8(&data, 2, 2).unwrap();

        // Check JXL signature
        assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);

        // Debug output
        eprintln!("Encoded black 2x2: {} bytes:", encoded.len());
        for (i, b) in encoded.iter().enumerate() {
            eprint!("{:02x} ", b);
            if (i + 1) % 16 == 0 {
                eprintln!();
            }
        }
        eprintln!();

        // Write to /tmp for testing with decoder
        std::fs::write("/tmp/test_black.jxl", &encoded).unwrap();
    }

    #[test]
    fn test_encode_white_2x2() {
        // 2x2 all-white image - tests single non-zero symbol encoding
        let data = vec![255u8; 2 * 2 * 3]; // All 255

        let encoded = encode_rgb8(&data, 2, 2).unwrap();
        assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);

        eprintln!("Encoded white 2x2: {} bytes:", encoded.len());
        for (i, b) in encoded.iter().enumerate() {
            eprint!("{:02x} ", b);
            if (i + 1) % 16 == 0 {
                eprintln!();
            }
        }
        eprintln!();

        std::fs::write("/tmp/test_white.jxl", &encoded).unwrap();
    }

    #[test]
    fn test_encoder_options() {
        let options = EncoderOptions::lossless();
        assert_eq!(options.distance, 0.0);

        let options = EncoderOptions::lossy(1.0);
        assert_eq!(options.distance, 1.0);
    }
}

#[cfg(test)]
mod gray_tests {
    use super::*;

    #[test]
    fn test_encode_gray_2x2() {
        // 2x2 grayscale with varied values
        let data = vec![0u8, 128, 64, 255];

        let encoded = Encoder::new().encode_gray8(&data, 2, 2).unwrap();

        // Check JXL signature
        assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);

        // Debug output
        eprintln!("Encoded gray 2x2: {} bytes:", encoded.len());
        for (i, b) in encoded.iter().enumerate() {
            eprint!("{:02x} ", b);
            if (i + 1) % 16 == 0 {
                eprintln!();
            }
        }
        eprintln!();

        std::fs::write("/tmp/test_gray.jxl", &encoded).unwrap();
    }
}

#[test]
fn test_encode_gray_binary() {
    // 2x2 grayscale with only 0 and 255
    let data = vec![0u8, 255, 0, 255];

    let encoded = Encoder::new().encode_gray8(&data, 2, 2).unwrap();
    std::fs::write("/tmp/test_gray_binary.jxl", &encoded).unwrap();

    eprintln!("Encoded gray binary 2x2: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_uniform_128() {
    // 2x2 grayscale all 128
    let data = vec![128u8; 4];

    let encoded = Encoder::new().encode_gray8(&data, 2, 2).unwrap();
    std::fs::write("/tmp/test_gray_128.jxl", &encoded).unwrap();
    eprintln!("Encoded gray 128 uniform: {} bytes", encoded.len());
}

#[test]
fn test_encode_rgb_simulated_gray() {
    // 2x2 RGB where each pixel has same R=G=B (simulating grayscale)
    let data = vec![
        0, 0, 0, // pixel (0,0) = black
        255, 255, 255, // pixel (1,0) = white
        0, 0, 0, // pixel (0,1) = black
        255, 255, 255, // pixel (1,1) = white
    ];

    let encoded = encode_rgb8(&data, 2, 2).unwrap();
    std::fs::write("/tmp/test_rgb_gray.jxl", &encoded).unwrap();
    eprintln!("Encoded RGB simulated gray: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_0_and_1() {
    // 2x2 grayscale with only 0 and 1
    let data = vec![0u8, 1, 0, 1];

    let encoded = Encoder::new().encode_gray8(&data, 2, 2).unwrap();
    std::fs::write("/tmp/test_gray_01.jxl", &encoded).unwrap();
    eprintln!("Encoded gray 0/1 2x2: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_0_and_3() {
    // 2x2 grayscale with 0 and 3 (zigzag: 0, 6)
    let data = vec![0u8, 3, 0, 3];

    let encoded = Encoder::new().encode_gray8(&data, 2, 2).unwrap();
    std::fs::write("/tmp/test_gray_03.jxl", &encoded).unwrap();
    eprintln!("Encoded gray 0/3 2x2: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_0_and_7() {
    // 2x2 grayscale with 0 and 7 (zigzag: 0, 14)
    let data = vec![0u8, 7, 0, 7];

    let encoded = Encoder::new().encode_gray8(&data, 2, 2).unwrap();
    std::fs::write("/tmp/test_gray_07.jxl", &encoded).unwrap();
    eprintln!("Encoded gray 0/7 2x2: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_0_and_15() {
    // 2x2 grayscale with 0 and 15 (zigzag: 0, 30)
    let data = vec![0u8, 15, 0, 15];

    let encoded = Encoder::new().encode_gray8(&data, 2, 2).unwrap();
    std::fs::write("/tmp/test_gray_015.jxl", &encoded).unwrap();
    eprintln!("Encoded gray 0/15 2x2: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_0_and_4() {
    // 2x2 grayscale with 0 and 4 (zigzag: 0, 8) - boundary: al_size=9, max_bits=4
    let data = vec![0u8, 4, 0, 4];

    let encoded = Encoder::new().encode_gray8(&data, 2, 2).unwrap();
    std::fs::write("/tmp/test_gray_04.jxl", &encoded).unwrap();
    eprintln!("Encoded gray 0/4 2x2: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_1_and_2() {
    // 2x2 grayscale with 1 and 2 (zigzag: 2, 4) - al_size=5, max_bits=3
    let data = vec![1u8, 2, 1, 2];

    let encoded = Encoder::new().encode_gray8(&data, 2, 2).unwrap();
    std::fs::write("/tmp/test_gray_12.jxl", &encoded).unwrap();
    eprintln!("Encoded gray 1/2 2x2: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_4x4_pattern() {
    // 4x4 grayscale with 4 unique values (max for simple Huffman)
    let data: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];

    let encoded = Encoder::new().encode_gray8(&data, 4, 4).unwrap();
    std::fs::write("/tmp/test_gray_4x4.jxl", &encoded).unwrap();
    eprintln!("Encoded gray 4x4 pattern: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_16_symbols() {
    // 4x4 gradient with 16 unique values - now works with full Huffman
    let data: Vec<u8> = (0u8..16).collect();

    let encoded = Encoder::new().encode_gray8(&data, 4, 4).unwrap();
    std::fs::write("/tmp/test_gray_16sym.jxl", &encoded).unwrap();
    eprintln!("Encoded 16-symbol gray 4x4: {} bytes", encoded.len());

    // Check JXL signature
    assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);
}

#[test]
fn test_encode_gray_256_symbols() {
    // 16x16 gradient with 256 unique values
    let data: Vec<u8> = (0u8..=255).collect();

    let encoded = Encoder::new().encode_gray8(&data, 16, 16).unwrap();
    std::fs::write("/tmp/test_gray_256sym.jxl", &encoded).unwrap();
    eprintln!("Encoded 256-symbol gray 16x16: {} bytes", encoded.len());

    // Check JXL signature
    assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);
}

#[test]
fn test_encode_gray_8x8_pattern() {
    // 8x8 grayscale checkerboard pattern
    let mut data = vec![0u8; 64];
    for y in 0..8 {
        for x in 0..8 {
            data[y * 8 + x] = if (x + y) % 2 == 0 { 0 } else { 128 };
        }
    }

    let encoded = Encoder::new().encode_gray8(&data, 8, 8).unwrap();
    std::fs::write("/tmp/test_gray_8x8.jxl", &encoded).unwrap();
    eprintln!("Encoded gray 8x8 checkerboard: {} bytes", encoded.len());
}
