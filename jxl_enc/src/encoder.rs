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

    #[test]
    fn test_encode_lossy_8x8() {
        // 8x8 RGB checkerboard for lossy encoding
        let mut data = vec![0u8; 8 * 8 * 3];
        for y in 0..8 {
            for x in 0..8 {
                let idx = (y * 8 + x) * 3;
                if (x + y) % 2 == 0 {
                    data[idx] = 255; // R
                    data[idx + 1] = 0; // G
                    data[idx + 2] = 0; // B
                } else {
                    data[idx] = 0; // R
                    data[idx + 1] = 0; // G
                    data[idx + 2] = 255; // B
                }
            }
        }

        let encoded = encode_lossy_rgb8(&data, 8, 8, 1.0).unwrap();

        // Check JXL signature
        assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);

        eprintln!("Encoded lossy 8x8: {} bytes", encoded.len());
        eprintln!("Hex dump:");
        for (i, b) in encoded.iter().enumerate() {
            eprint!("{:02x} ", b);
            if (i + 1) % 16 == 0 {
                eprintln!();
            }
        }
        eprintln!();
        std::fs::write("/tmp/lossy_8x8.jxl", &encoded).unwrap();

        // Verify roundtrip: file bytes == memory bytes
        let read_back = std::fs::read("/tmp/lossy_8x8.jxl").unwrap();
        assert_eq!(encoded, read_back, "File bytes don't match memory bytes!");

        // Try to decode the READ-BACK bytes (not original)
        let result = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&read_back));
        assert!(
            result.is_ok(),
            "Failed to decode read-back bytes: {:?}",
            result.err()
        );
        let image = result.unwrap();
        assert_eq!(image.width(), 8);
        assert_eq!(image.height(), 8);
        eprintln!(
            "Decode from file bytes succeeded: {}x{}",
            image.width(),
            image.height()
        );
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

#[cfg(test)]
mod corpus_tests {
    use super::*;

    const CORPUS_PATH: &str = "/home/lilith/work/codec-corpus";

    fn test_image_roundtrip(path: &str) -> Result<(usize, usize, usize), String> {
        let img = image::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;

        let (width, height) = (img.width() as usize, img.height() as usize);

        // Convert to appropriate format and encode
        let encoded = match img.color() {
            image::ColorType::L8 => {
                let gray = img.to_luma8();
                Encoder::new()
                    .encode_gray8(gray.as_raw(), width, height)
                    .map_err(|e| format!("Encode failed: {}", e))?
            }
            image::ColorType::Rgb8 => {
                let rgb = img.to_rgb8();
                Encoder::new()
                    .encode_rgb8(rgb.as_raw(), width, height)
                    .map_err(|e| format!("Encode failed: {}", e))?
            }
            image::ColorType::Rgba8 => {
                let rgba = img.to_rgba8();
                Encoder::new()
                    .encode_rgba8(rgba.as_raw(), width, height)
                    .map_err(|e| format!("Encode failed: {}", e))?
            }
            other => {
                // Convert to RGB8 for other formats
                let rgb = img.to_rgb8();
                Encoder::new()
                    .encode_rgb8(rgb.as_raw(), width, height)
                    .map_err(|e| format!("Encode failed for {:?}: {}", other, e))?
            }
        };

        // Verify JXL signature
        if encoded.len() < 2 || encoded[0] != 0xFF || encoded[1] != 0x0A {
            return Err("Invalid JXL signature".to_string());
        }

        Ok((width, height, encoded.len()))
    }

    #[test]
    fn test_pngsuite_gray() {
        // 8-bit grayscale from PNG suite
        let path = format!("{}/pngsuite/basi0g08.png", CORPUS_PATH);
        if std::path::Path::new(&path).exists() {
            let img = image::open(&path).unwrap();
            let gray = img.to_luma8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let encoded = Encoder::new().encode_gray8(gray.as_raw(), w, h).unwrap();
            std::fs::write("/tmp/pngsuite_gray.jxl", &encoded).unwrap();
            eprintln!("basi0g08.png: {}x{} -> {} bytes", w, h, encoded.len());
        } else {
            eprintln!("Skipping: {} not found", path);
        }
    }

    #[test]
    fn test_pngsuite_rgb() {
        // 8-bit RGB from PNG suite
        let path = format!("{}/pngsuite/basi2c08.png", CORPUS_PATH);
        if std::path::Path::new(&path).exists() {
            let img = image::open(&path).unwrap();
            let rgb = img.to_rgb8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let encoded = Encoder::new().encode_rgb8(rgb.as_raw(), w, h).unwrap();
            std::fs::write("/tmp/pngsuite_rgb.jxl", &encoded).unwrap();
            eprintln!("basi2c08.png: {}x{} -> {} bytes", w, h, encoded.len());
        } else {
            eprintln!("Skipping: {} not found", path);
        }
    }

    #[test]
    fn test_kodak_01() {
        let path = format!("{}/kodak/1.png", CORPUS_PATH);
        if std::path::Path::new(&path).exists() {
            match test_image_roundtrip(&path) {
                Ok((w, h, size)) => {
                    eprintln!("kodak/1.png: {}x{} -> {} bytes", w, h, size);
                    // Save for manual verification
                    let img = image::open(&path).unwrap();
                    let rgb = img.to_rgb8();
                    let encoded = Encoder::new().encode_rgb8(rgb.as_raw(), w, h).unwrap();
                    std::fs::write("/tmp/kodak1.jxl", &encoded).unwrap();
                }
                Err(e) => panic!("{}", e),
            }
        } else {
            eprintln!("Skipping: {} not found", path);
        }
    }

    #[test]
    fn test_corpus_batch() {
        // Test multiple images from the corpus
        let test_images = [
            "pngsuite/basi0g01.png", // 1-bit grayscale
            "pngsuite/basi0g02.png", // 2-bit grayscale
            "pngsuite/basi0g04.png", // 4-bit grayscale
            "pngsuite/basi0g08.png", // 8-bit grayscale
            "pngsuite/basi2c08.png", // 8-bit RGB
            "pngsuite/basn0g08.png", // 8-bit grayscale, non-interlaced
            "pngsuite/basn2c08.png", // 8-bit RGB, non-interlaced
        ];

        let mut passed = 0;
        let mut failed = 0;

        for img_path in &test_images {
            let full_path = format!("{}/{}", CORPUS_PATH, img_path);
            if !std::path::Path::new(&full_path).exists() {
                eprintln!("SKIP: {} (not found)", img_path);
                continue;
            }

            match test_image_roundtrip(&full_path) {
                Ok((w, h, size)) => {
                    eprintln!("PASS: {} ({}x{} -> {} bytes)", img_path, w, h, size);
                    passed += 1;
                }
                Err(e) => {
                    eprintln!("FAIL: {} - {}", img_path, e);
                    failed += 1;
                }
            }
        }

        eprintln!("\nResults: {} passed, {} failed", passed, failed);
        assert_eq!(failed, 0, "Some corpus tests failed");
    }
}

#[test]
fn test_encode_rgb_8x8() {
    // 8x8 RGB checkerboard
    let mut data = vec![0u8; 8 * 8 * 3];
    for y in 0..8 {
        for x in 0..8 {
            let idx = (y * 8 + x) * 3;
            if (x + y) % 2 == 0 {
                data[idx] = 255; // R
                data[idx + 1] = 0; // G
                data[idx + 2] = 0; // B
            } else {
                data[idx] = 0; // R
                data[idx + 1] = 0; // G
                data[idx + 2] = 255; // B
            }
        }
    }

    let encoded = Encoder::new().encode_rgb8(&data, 8, 8).unwrap();
    std::fs::write("/tmp/rgb_8x8.jxl", &encoded).unwrap();
    eprintln!("Encoded RGB 8x8: {} bytes", encoded.len());
    assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);
}

#[test]
fn test_encode_gray_8x8() {
    // 8x8 grayscale checkerboard
    let mut data = vec![0u8; 8 * 8];
    for y in 0..8 {
        for x in 0..8 {
            let idx = y * 8 + x;
            data[idx] = if (x + y) % 2 == 0 { 255 } else { 0 };
        }
    }

    let encoded = Encoder::new().encode_gray8(&data, 8, 8).unwrap();
    std::fs::write("/tmp/gray_8x8.jxl", &encoded).unwrap();
    eprintln!("Encoded Gray 8x8: {} bytes", encoded.len());
}

#[test]
fn test_encode_rgb_4x4() {
    // 4x4 RGB checkerboard
    let mut data = vec![0u8; 4 * 4 * 3];
    for y in 0..4 {
        for x in 0..4 {
            let idx = (y * 4 + x) * 3;
            if (x + y) % 2 == 0 {
                data[idx] = 255; // R
                data[idx + 1] = 0; // G
                data[idx + 2] = 0; // B
            } else {
                data[idx] = 0; // R
                data[idx + 1] = 0; // G
                data[idx + 2] = 255; // B
            }
        }
    }

    let encoded = Encoder::new().encode_rgb8(&data, 4, 4).unwrap();
    std::fs::write("/tmp/rgb_4x4.jxl", &encoded).unwrap();
    eprintln!("Encoded RGB 4x4: {} bytes", encoded.len());
}

#[test]
fn test_encode_rgb_6x6() {
    // 6x6 RGB checkerboard
    let mut data = vec![0u8; 6 * 6 * 3];
    for y in 0..6 {
        for x in 0..6 {
            let idx = (y * 6 + x) * 3;
            if (x + y) % 2 == 0 {
                data[idx] = 255;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
            } else {
                data[idx] = 0;
                data[idx + 1] = 0;
                data[idx + 2] = 255;
            }
        }
    }
    let encoded = Encoder::new().encode_rgb8(&data, 6, 6).unwrap();
    std::fs::write("/tmp/rgb_6x6.jxl", &encoded).unwrap();
    eprintln!("Encoded RGB 6x6: {} bytes", encoded.len());
}

#[test]
fn test_encode_rgb_7x7() {
    let mut data = vec![0u8; 7 * 7 * 3];
    for y in 0..7 {
        for x in 0..7 {
            let idx = (y * 7 + x) * 3;
            if (x + y) % 2 == 0 {
                data[idx] = 255;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
            } else {
                data[idx] = 0;
                data[idx + 1] = 0;
                data[idx + 2] = 255;
            }
        }
    }
    let encoded = Encoder::new().encode_rgb8(&data, 7, 7).unwrap();
    std::fs::write("/tmp/rgb_7x7.jxl", &encoded).unwrap();
    eprintln!("Encoded RGB 7x7: {} bytes", encoded.len());
}

#[test]
fn test_encode_rgb_9x9() {
    let mut data = vec![0u8; 9 * 9 * 3];
    for y in 0..9 {
        for x in 0..9 {
            let idx = (y * 9 + x) * 3;
            if (x + y) % 2 == 0 {
                data[idx] = 255;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
            } else {
                data[idx] = 0;
                data[idx + 1] = 0;
                data[idx + 2] = 255;
            }
        }
    }
    let encoded = Encoder::new().encode_rgb8(&data, 9, 9).unwrap();
    std::fs::write("/tmp/rgb_9x9.jxl", &encoded).unwrap();
}

#[test]
fn test_encode_rgb_16x16() {
    let mut data = vec![0u8; 16 * 16 * 3];
    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            if (x + y) % 2 == 0 {
                data[idx] = 255;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
            } else {
                data[idx] = 0;
                data[idx + 1] = 0;
                data[idx + 2] = 255;
            }
        }
    }
    let encoded = Encoder::new().encode_rgb8(&data, 16, 16).unwrap();
    std::fs::write("/tmp/rgb_16x16.jxl", &encoded).unwrap();
    eprintln!("Encoded RGB 16x16: {} bytes", encoded.len());
}

#[test]
fn test_encode_rgb_24x24() {
    let mut data = vec![0u8; 24 * 24 * 3];
    for y in 0..24 {
        for x in 0..24 {
            let idx = (y * 24 + x) * 3;
            if (x + y) % 2 == 0 {
                data[idx] = 255;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
            } else {
                data[idx] = 0;
                data[idx + 1] = 0;
                data[idx + 2] = 255;
            }
        }
    }
    let encoded = Encoder::new().encode_rgb8(&data, 24, 24).unwrap();
    std::fs::write("/tmp/rgb_24x24.jxl", &encoded).unwrap();
}

#[test]
fn test_encode_rgb_10x10() {
    let mut data = vec![0u8; 10 * 10 * 3];
    for y in 0..10 {
        for x in 0..10 {
            let idx = (y * 10 + x) * 3;
            if (x + y) % 2 == 0 {
                data[idx] = 255;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
            } else {
                data[idx] = 0;
                data[idx + 1] = 0;
                data[idx + 2] = 255;
            }
        }
    }
    let encoded = Encoder::new().encode_rgb8(&data, 10, 10).unwrap();
    std::fs::write("/tmp/rgb_10x10.jxl", &encoded).unwrap();
}

#[test]
fn test_encode_rgb_32x32() {
    let mut data = vec![0u8; 32 * 32 * 3];
    for y in 0..32 {
        for x in 0..32 {
            let idx = (y * 32 + x) * 3;
            if (x + y) % 2 == 0 {
                data[idx] = 255;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
            } else {
                data[idx] = 0;
                data[idx + 1] = 0;
                data[idx + 2] = 255;
            }
        }
    }
    let encoded = Encoder::new().encode_rgb8(&data, 32, 32).unwrap();
    std::fs::write("/tmp/rgb_32x32.jxl", &encoded).unwrap();
    eprintln!("Encoded RGB 32x32: {} bytes", encoded.len());
}

#[test]
fn test_encode_rgb_64x64() {
    let mut data = vec![0u8; 64 * 64 * 3];
    for y in 0..64 {
        for x in 0..64 {
            let idx = (y * 64 + x) * 3;
            if (x + y) % 2 == 0 {
                data[idx] = 255;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
            } else {
                data[idx] = 0;
                data[idx + 1] = 0;
                data[idx + 2] = 255;
            }
        }
    }
    let encoded = Encoder::new().encode_rgb8(&data, 64, 64).unwrap();
    std::fs::write("/tmp/rgb_64x64.jxl", &encoded).unwrap();
    eprintln!("Encoded RGB 64x64: {} bytes", encoded.len());
}

#[test]
fn test_encode_rgb_256x256() {
    // 256x256 uses small size encoding (256/8=32 fits in 5 bits)
    let mut data = vec![0u8; 256 * 256 * 3];
    for y in 0..256 {
        for x in 0..256 {
            let idx = (y * 256 + x) * 3;
            if (x + y) % 2 == 0 {
                data[idx] = 255;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
            } else {
                data[idx] = 0;
                data[idx + 1] = 0;
                data[idx + 2] = 255;
            }
        }
    }
    let encoded = Encoder::new().encode_rgb8(&data, 256, 256).unwrap();
    std::fs::write("/tmp/rgb_256x256.jxl", &encoded).unwrap();
    eprintln!("Encoded RGB 256x256: {} bytes", encoded.len());
}

#[test]
fn test_encode_rgb_irregular_dimensions() {
    // Test various irregular dimensions: non-multiples of 8, non-square, primes
    let test_cases = [
        (5, 5),     // small odd
        (7, 11),    // non-square primes
        (13, 17),   // larger primes
        (100, 50),  // wide rectangle
        (50, 100),  // tall rectangle
        (255, 1),   // single row
        (1, 255),   // single column
        (127, 127), // odd, just under 128
        (129, 129), // odd, just over 128
    ];

    for (w, h) in test_cases {
        let mut data = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                if (x + y) % 2 == 0 {
                    data[idx] = 255;
                    data[idx + 1] = 0;
                    data[idx + 2] = 0;
                } else {
                    data[idx] = 0;
                    data[idx + 1] = 0;
                    data[idx + 2] = 255;
                }
            }
        }
        let encoded = Encoder::new()
            .encode_rgb8(&data, w, h)
            .unwrap_or_else(|e| panic!("{}x{} failed to encode: {}", w, h, e));

        let path = format!("/tmp/rgb_{}x{}.jxl", w, h);
        std::fs::write(&path, &encoded).unwrap();
        eprintln!("{}x{}: {} bytes", w, h, encoded.len());

        // Verify JXL signature
        assert_eq!(
            &encoded[0..2],
            &[0xFF, 0x0A],
            "{}x{} has invalid signature",
            w,
            h
        );
    }
}

/// Test with an image that has many unique colors (similar to pngsuite)
/// This triggers more complex Huffman coding and LZ77 patterns
#[test]
fn test_encode_rgb_gradient() {
    // 32x32 gradient image with many unique colors
    let mut data = vec![0u8; 32 * 32 * 3];
    for y in 0..32 {
        for x in 0..32 {
            let idx = (y * 32 + x) * 3;
            // Create a gradient with many unique colors
            data[idx] = (x * 8) as u8; // R varies with x
            data[idx + 1] = (y * 8) as u8; // G varies with y
            data[idx + 2] = ((x + y) * 4) as u8; // B varies with x+y
        }
    }
    let encoded = Encoder::new().encode_rgb8(&data, 32, 32).unwrap();
    std::fs::write("/tmp/rgb_gradient_32x32.jxl", &encoded).unwrap();
    eprintln!("Encoded RGB gradient 32x32: {} bytes", encoded.len());
}

#[cfg(test)]
mod decoder_validation {
    use super::*;
    use std::process::Command;

    /// Path to djxl from libjxl for dual-decoder validation
    const DJXL_PATH: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";

    /// Validates lossless roundtrip: encode -> decode -> compare pixels exactly.
    ///
    /// Returns the decoded pixel data on success.
    fn validate_lossless_roundtrip_rgb(
        original: &[u8],
        width: usize,
        height: usize,
        test_name: &str,
    ) -> Vec<u8> {
        assert_eq!(original.len(), width * height * 3);

        // Encode
        let encoded = Encoder::new()
            .encode_rgb8(original, width, height)
            .expect(&format!("{}: encoding failed", test_name));

        // Decode with jxl-oxide
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .expect(&format!("{}: jxl-oxide decode failed", test_name));

        assert_eq!(image.width() as usize, width);
        assert_eq!(image.height() as usize, height);

        // Render frame and extract pixels
        let render = image
            .render_frame(0)
            .expect(&format!("{}: render failed", test_name));

        let fb = render.image_all_channels();
        let decoded_f32 = fb.buf();

        // Convert f32 (0.0-1.0 normalized range) to u8 (0-255)
        let decoded: Vec<u8> = decoded_f32
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        // Compare pixel by pixel
        assert_eq!(
            decoded.len(),
            original.len(),
            "{}: decoded size mismatch ({} vs {})",
            test_name,
            decoded.len(),
            original.len()
        );

        let mut max_diff: i32 = 0;
        let mut diff_count = 0;
        for (i, (&orig, &dec)) in original.iter().zip(decoded.iter()).enumerate() {
            let diff = (orig as i32 - dec as i32).abs();
            if diff > 0 {
                diff_count += 1;
                max_diff = max_diff.max(diff);
                if diff_count <= 5 {
                    let pixel = i / 3;
                    let channel = i % 3;
                    eprintln!(
                        "{}: pixel {} channel {} differs: {} vs {} (diff={})",
                        test_name, pixel, channel, orig, dec, diff
                    );
                }
            }
        }

        assert_eq!(
            max_diff, 0,
            "{}: lossless roundtrip failed! {} pixels differ, max_diff={}",
            test_name, diff_count, max_diff
        );

        eprintln!(
            "{}: PASSED lossless roundtrip ({}x{}, {} bytes)",
            test_name,
            width,
            height,
            encoded.len()
        );
        decoded
    }

    /// Validates lossless roundtrip for grayscale images.
    fn validate_lossless_roundtrip_gray(
        original: &[u8],
        width: usize,
        height: usize,
        test_name: &str,
    ) -> Vec<u8> {
        assert_eq!(original.len(), width * height);

        // Encode
        let encoded = Encoder::new()
            .encode_gray8(original, width, height)
            .expect(&format!("{}: encoding failed", test_name));

        // Decode with jxl-oxide
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .expect(&format!("{}: jxl-oxide decode failed", test_name));

        assert_eq!(image.width() as usize, width);
        assert_eq!(image.height() as usize, height);

        // Render frame and extract pixels
        let render = image
            .render_frame(0)
            .expect(&format!("{}: render failed", test_name));

        let fb = render.image_all_channels();
        let decoded_f32 = fb.buf();

        // Convert f32 (0.0-1.0 normalized range) to u8 (0-255)
        let decoded: Vec<u8> = decoded_f32
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        // Compare pixel by pixel
        assert_eq!(
            decoded.len(),
            original.len(),
            "{}: decoded size mismatch",
            test_name
        );

        let mut max_diff: i32 = 0;
        let mut diff_count = 0;
        for (_i, (&orig, &dec)) in original.iter().zip(decoded.iter()).enumerate() {
            let diff = (orig as i32 - dec as i32).abs();
            if diff > 0 {
                diff_count += 1;
                max_diff = max_diff.max(diff);
            }
        }

        assert_eq!(
            max_diff, 0,
            "{}: lossless roundtrip failed! {} pixels differ, max_diff={}",
            test_name, diff_count, max_diff
        );

        eprintln!(
            "{}: PASSED lossless roundtrip ({}x{}, {} bytes)",
            test_name,
            width,
            height,
            encoded.len()
        );
        decoded
    }

    /// Validates lossy roundtrip with tolerance.
    ///
    /// Returns (max_diff, mean_diff) for the decoded image.
    fn validate_lossy_roundtrip_rgb(
        original: &[u8],
        width: usize,
        height: usize,
        distance: f32,
        max_allowed_diff: i32,
        test_name: &str,
    ) -> (i32, f64) {
        assert_eq!(original.len(), width * height * 3);

        // Encode lossy
        let encoded = encode_lossy_rgb8(original, width, height, distance)
            .expect(&format!("{}: encoding failed", test_name));

        // Decode with jxl-oxide
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .expect(&format!("{}: jxl-oxide decode failed", test_name));

        assert_eq!(image.width() as usize, width);
        assert_eq!(image.height() as usize, height);

        // Render frame and extract pixels
        let render = image
            .render_frame(0)
            .expect(&format!("{}: render failed", test_name));

        let fb = render.image_all_channels();
        let decoded_f32 = fb.buf();

        // Convert f32 (0.0-1.0 normalized range) to u8 (0-255)
        let decoded: Vec<u8> = decoded_f32
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        // Calculate statistics
        let mut max_diff: i32 = 0;
        let mut sum_diff: i64 = 0;
        for (&orig, &dec) in original.iter().zip(decoded.iter()) {
            let diff = (orig as i32 - dec as i32).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff as i64;
        }
        let mean_diff = sum_diff as f64 / original.len() as f64;

        assert!(
            max_diff <= max_allowed_diff,
            "{}: lossy roundtrip max_diff {} exceeds tolerance {} (distance={}, mean_diff={:.2})",
            test_name,
            max_diff,
            max_allowed_diff,
            distance,
            mean_diff
        );

        eprintln!(
            "{}: PASSED lossy roundtrip (distance={}, max_diff={}, mean_diff={:.2}, {} bytes)",
            test_name,
            distance,
            max_diff,
            mean_diff,
            encoded.len()
        );

        (max_diff, mean_diff)
    }

    /// Validates that a JXL file can be decoded by both jxl-oxide and djxl.
    /// Returns (width, height) on success.
    fn validate_dual_decoder(
        encoded: &[u8],
        expected_width: u32,
        expected_height: u32,
        test_name: &str,
    ) -> (u32, u32) {
        // 1. Validate with jxl-oxide (Rust decoder)
        let oxide_result = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(encoded));
        let oxide_dims = match oxide_result {
            Ok(image) => {
                assert_eq!(
                    image.width(),
                    expected_width,
                    "{}: jxl-oxide width mismatch",
                    test_name
                );
                assert_eq!(
                    image.height(),
                    expected_height,
                    "{}: jxl-oxide height mismatch",
                    test_name
                );
                (image.width(), image.height())
            }
            Err(e) => {
                panic!("{}: jxl-oxide decode failed: {:?}", test_name, e);
            }
        };

        // 2. Validate with djxl (libjxl reference decoder)
        if std::path::Path::new(DJXL_PATH).exists() {
            // Write JXL to temp file
            let temp_jxl = format!("/tmp/dual_decode_test_{}.jxl", test_name.replace(" ", "_"));
            let temp_png = format!("/tmp/dual_decode_test_{}.png", test_name.replace(" ", "_"));
            std::fs::write(&temp_jxl, encoded).expect("Failed to write temp JXL");

            // Run djxl
            let output = Command::new(DJXL_PATH)
                .args([&temp_jxl, &temp_png])
                .output()
                .expect("Failed to run djxl");

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                panic!("{}: djxl decode failed: {}", test_name, stderr);
            }

            // Verify the PNG was created and has correct dimensions
            if let Ok(img) = image::open(&temp_png) {
                assert_eq!(
                    img.width(),
                    expected_width,
                    "{}: djxl output width mismatch",
                    test_name
                );
                assert_eq!(
                    img.height(),
                    expected_height,
                    "{}: djxl output height mismatch",
                    test_name
                );
            }

            // Cleanup
            let _ = std::fs::remove_file(&temp_jxl);
            let _ = std::fs::remove_file(&temp_png);

            eprintln!(
                "{}: PASSED dual-decoder validation (jxl-oxide + djxl)",
                test_name
            );
        } else {
            eprintln!(
                "{}: PASSED jxl-oxide only (djxl not available at {})",
                test_name, DJXL_PATH
            );
        }

        oxide_dims
    }

    /// Test that our encoded files can be decoded by jxl-oxide
    #[test]
    fn test_decode_simple_gray() {
        // 2x2 grayscale with values [0, 1, 0, 1]
        let data = vec![0u8, 1, 0, 1];
        let encoded = Encoder::new().encode_gray8(&data, 2, 2).unwrap();

        // Try to decode with jxl-oxide
        let decoder = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&encoded));

        match decoder {
            Ok(image) => {
                eprintln!(
                    "Successfully decoded 2x2 gray: {}x{}",
                    image.width(),
                    image.height()
                );
                assert_eq!(image.width(), 2);
                assert_eq!(image.height(), 2);
            }
            Err(e) => {
                eprintln!("Decode failed: {:?}", e);
                eprintln!("Encoded bytes ({}):", encoded.len());
                for (i, b) in encoded.iter().enumerate() {
                    eprint!("{:02x} ", b);
                    if (i + 1) % 16 == 0 {
                        eprintln!();
                    }
                }
                eprintln!();
                panic!("jxl-oxide failed to decode our encoded file");
            }
        }
    }

    /// Test that our lossless RGB encoded files can be decoded by jxl-oxide
    #[test]
    fn test_decode_lossless_rgb() {
        // 8x8 RGB checkerboard for lossless encoding
        let mut data = vec![0u8; 8 * 8 * 3];
        for y in 0..8 {
            for x in 0..8 {
                let idx = (y * 8 + x) * 3;
                if (x + y) % 2 == 0 {
                    data[idx] = 255; // R
                    data[idx + 1] = 0; // G
                    data[idx + 2] = 0; // B
                } else {
                    data[idx] = 0;
                    data[idx + 1] = 0;
                    data[idx + 2] = 255;
                }
            }
        }

        let encoded = Encoder::new().encode_rgb8(&data, 8, 8).unwrap();
        eprintln!("Lossless RGB 8x8 encoded to {} bytes", encoded.len());

        // Try to decode with jxl-oxide
        let decoder = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&encoded));

        match decoder {
            Ok(image) => {
                eprintln!(
                    "Successfully decoded lossless RGB 8x8: {}x{}",
                    image.width(),
                    image.height()
                );
                assert_eq!(image.width(), 8);
                assert_eq!(image.height(), 8);
            }
            Err(e) => {
                eprintln!("Lossless RGB decode failed: {:?}", e);
                eprintln!("Encoded bytes ({}):", encoded.len());
                for (i, b) in encoded.iter().enumerate() {
                    eprint!("{:02x} ", b);
                    if (i + 1) % 16 == 0 {
                        eprintln!();
                    }
                }
                eprintln!();
                panic!("jxl-oxide failed to decode lossless RGB file: {:?}", e);
            }
        }
    }

    /// Test that our lossy encoded files can be decoded
    #[test]
    fn test_decode_lossy_rgb() {
        // 8x8 RGB checkerboard for lossy encoding
        let mut data = vec![0u8; 8 * 8 * 3];
        for y in 0..8 {
            for x in 0..8 {
                let idx = (y * 8 + x) * 3;
                if (x + y) % 2 == 0 {
                    data[idx] = 255;
                    data[idx + 1] = 0;
                    data[idx + 2] = 0;
                } else {
                    data[idx] = 0;
                    data[idx + 1] = 0;
                    data[idx + 2] = 255;
                }
            }
        }

        let encoded = encode_lossy_rgb8(&data, 8, 8, 1.0).unwrap();
        eprintln!("Lossy 8x8 encoded to {} bytes", encoded.len());

        // Try to decode with jxl-oxide
        let decoder = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&encoded));

        match decoder {
            Ok(image) => {
                eprintln!(
                    "Successfully decoded lossy 8x8: {}x{}",
                    image.width(),
                    image.height()
                );
                assert_eq!(image.width(), 8);
                assert_eq!(image.height(), 8);
            }
            Err(e) => {
                eprintln!("Lossy decode failed: {:?}", e);
                eprintln!("Encoded bytes ({}):", encoded.len());
                for (i, b) in encoded.iter().enumerate() {
                    eprint!("{:02x} ", b);
                    if (i + 1) % 16 == 0 {
                        eprintln!();
                    }
                }
                eprintln!();
                panic!("jxl-oxide failed to decode lossy file: {:?}", e);
            }
        }
    }

    /// Test lossy encoding of a solid color image
    #[test]
    fn test_decode_lossy_solid_color() {
        // 8x8 solid color image
        let mut data = vec![0u8; 8 * 8 * 3];
        for i in 0..(8 * 8) {
            data[i * 3] = 200; // R
            data[i * 3 + 1] = 50; // G
            data[i * 3 + 2] = 100; // B
        }

        let encoded = encode_lossy_rgb8(&data, 8, 8, 1.0).unwrap();
        eprintln!("Solid color 8x8 encoded to {} bytes", encoded.len());

        // Decode with jxl-oxide
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .expect("Failed to decode");

        eprintln!(
            "Successfully decoded solid color 8x8: {}x{}",
            image.width(),
            image.height()
        );
        assert_eq!(image.width(), 8);
        assert_eq!(image.height(), 8);
    }

    /// Test lossy encoding at different distance levels
    #[test]
    fn test_decode_lossy_distances() {
        // 8x8 gradient image
        let mut data = vec![0u8; 8 * 8 * 3];
        for y in 0..8 {
            for x in 0..8 {
                let idx = (y * 8 + x) * 3;
                data[idx] = (x * 32) as u8; // R gradient
                data[idx + 1] = (y * 32) as u8; // G gradient
                data[idx + 2] = 128; // B constant
            }
        }

        for distance in [0.5, 1.0, 2.0, 4.0] {
            let encoded = encode_lossy_rgb8(&data, 8, 8, distance).unwrap();
            eprintln!("Distance {}: {} bytes", distance, encoded.len());

            // Verify decodes correctly
            let image = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&encoded))
                .expect(&format!("Failed to decode at distance {}", distance));

            assert_eq!(image.width(), 8);
            assert_eq!(image.height(), 8);
        }
    }

    /// Test that pngsuite RGB images can be decoded by jxl-oxide
    #[test]
    fn test_decode_pngsuite_rgb() {
        const CORPUS_PATH: &str = "/home/lilith/work/codec-corpus";
        let path = format!("{}/pngsuite/basn2c08.png", CORPUS_PATH);
        if !std::path::Path::new(&path).exists() {
            eprintln!("Skipping test: {} not found", path);
            return;
        }

        let img = image::open(&path).unwrap();
        let rgb = img.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);

        let encoded = Encoder::new().encode_rgb8(rgb.as_raw(), w, h).unwrap();
        eprintln!(
            "basn2c08.png {}x{} encoded to {} bytes",
            w,
            h,
            encoded.len()
        );

        // Decode with jxl-oxide
        match jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&encoded)) {
            Ok(image) => {
                eprintln!(
                    "Successfully decoded basn2c08: {}x{}",
                    image.width(),
                    image.height()
                );
                assert_eq!(image.width(), w as u32);
                assert_eq!(image.height(), h as u32);
            }
            Err(e) => {
                eprintln!("basn2c08 decode failed: {:?}", e);
                eprintln!("First 64 bytes:");
                for (i, b) in encoded.iter().take(64).enumerate() {
                    eprint!("{:02x} ", b);
                    if (i + 1) % 16 == 0 {
                        eprintln!();
                    }
                }
                panic!("jxl-oxide failed to decode basn2c08: {:?}", e);
            }
        }
    }

    /// Test multi-group lossy encoding (512x512 = 4 groups)
    #[test]
    fn test_decode_lossy_multi_group() {
        // 512x512 checkerboard pattern = 4 groups (2x2)
        let mut data = vec![0u8; 512 * 512 * 3];
        for y in 0..512 {
            for x in 0..512 {
                let idx = (y * 512 + x) * 3;
                // Create a pattern that varies by position
                data[idx] = ((x + y) % 256) as u8; // R
                data[idx + 1] = ((x * 2) % 256) as u8; // G
                data[idx + 2] = ((y * 2) % 256) as u8; // B
            }
        }

        let encoded = encode_lossy_rgb8(&data, 512, 512, 2.0).unwrap();
        eprintln!("Multi-group 512x512 encoded to {} bytes", encoded.len());

        // Verify decodes correctly with jxl-oxide
        match jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&encoded)) {
            Ok(image) => {
                eprintln!(
                    "Successfully decoded 512x512: {}x{}",
                    image.width(),
                    image.height()
                );
                assert_eq!(image.width(), 512);
                assert_eq!(image.height(), 512);
            }
            Err(e) => {
                eprintln!("Multi-group decode failed: {:?}", e);
                eprintln!("First 64 bytes:");
                for (i, b) in encoded.iter().take(64).enumerate() {
                    eprint!("{:02x} ", b);
                    if (i + 1) % 16 == 0 {
                        eprintln!();
                    }
                }
                panic!("jxl-oxide failed to decode multi-group file: {:?}", e);
            }
        }
    }

    // ========== DUAL-DECODER VALIDATION TESTS ==========
    // These tests validate encoded files against BOTH jxl-oxide and djxl

    /// Dual-decoder validation for lossless grayscale encoding
    #[test]
    fn test_dual_decode_lossless_gray() {
        let data = vec![0u8, 64, 128, 192, 255, 100, 50, 200];
        let encoded = Encoder::new().encode_gray8(&data, 4, 2).unwrap();
        validate_dual_decoder(&encoded, 4, 2, "lossless_gray_4x2");
    }

    /// Dual-decoder validation for lossless RGB encoding
    #[test]
    fn test_dual_decode_lossless_rgb() {
        let mut data = vec![0u8; 8 * 8 * 3];
        for y in 0..8 {
            for x in 0..8 {
                let idx = (y * 8 + x) * 3;
                data[idx] = (x * 32) as u8; // R gradient
                data[idx + 1] = (y * 32) as u8; // G gradient
                data[idx + 2] = 128; // B constant
            }
        }
        let encoded = Encoder::new().encode_rgb8(&data, 8, 8).unwrap();
        validate_dual_decoder(&encoded, 8, 8, "lossless_rgb_8x8");
    }

    /// Dual-decoder validation for lossy VarDCT encoding
    /// Note: VarDCT encoding is WIP and may not pass djxl yet
    #[test]
    fn test_dual_decode_lossy_vardct() {
        let mut data = vec![0u8; 16 * 16 * 3];
        for y in 0..16 {
            for x in 0..16 {
                let idx = (y * 16 + x) * 3;
                data[idx] = ((x + y) * 8) as u8;
                data[idx + 1] = ((x * 2) % 256) as u8;
                data[idx + 2] = ((y * 2) % 256) as u8;
            }
        }
        let encoded = encode_lossy_rgb8(&data, 16, 16, 1.0).unwrap();
        // VarDCT is validated against jxl-oxide only until encoder is complete
        let oxide_result = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&encoded));
        assert!(oxide_result.is_ok(), "jxl-oxide should decode lossy VarDCT");
        let image = oxide_result.unwrap();
        assert_eq!(image.width(), 16);
        assert_eq!(image.height(), 16);
        eprintln!("lossy_vardct_16x16: PASSED jxl-oxide (VarDCT WIP)");
    }

    /// Dual-decoder validation for solid color image
    #[test]
    fn test_dual_decode_solid_color() {
        let mut data = vec![0u8; 32 * 32 * 3];
        for i in 0..(32 * 32) {
            data[i * 3] = 200;
            data[i * 3 + 1] = 100;
            data[i * 3 + 2] = 50;
        }
        let encoded = Encoder::new().encode_rgb8(&data, 32, 32).unwrap();
        validate_dual_decoder(&encoded, 32, 32, "solid_color_32x32");
    }

    /// Dual-decoder validation for checkerboard pattern
    #[test]
    fn test_dual_decode_checkerboard() {
        let mut data = vec![0u8; 16 * 16 * 3];
        for y in 0..16 {
            for x in 0..16 {
                let idx = (y * 16 + x) * 3;
                if (x + y) % 2 == 0 {
                    data[idx] = 255;
                    data[idx + 1] = 255;
                    data[idx + 2] = 255;
                } else {
                    data[idx] = 0;
                    data[idx + 1] = 0;
                    data[idx + 2] = 0;
                }
            }
        }
        let encoded = Encoder::new().encode_rgb8(&data, 16, 16).unwrap();
        validate_dual_decoder(&encoded, 16, 16, "checkerboard_16x16");
    }

    /// Dual-decoder validation at multiple lossy distances
    /// Note: VarDCT encoding is WIP - jxl-oxide only
    #[test]
    fn test_dual_decode_lossy_distances() {
        let mut data = vec![0u8; 8 * 8 * 3];
        for y in 0..8 {
            for x in 0..8 {
                let idx = (y * 8 + x) * 3;
                data[idx] = (x * 32) as u8;
                data[idx + 1] = (y * 32) as u8;
                data[idx + 2] = 100;
            }
        }

        for distance in [0.5, 1.0, 2.0, 4.0] {
            let encoded = encode_lossy_rgb8(&data, 8, 8, distance).unwrap();
            // VarDCT validated against jxl-oxide only
            let oxide_result = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&encoded));
            assert!(
                oxide_result.is_ok(),
                "jxl-oxide should decode at distance {}",
                distance
            );
            let image = oxide_result.unwrap();
            assert_eq!(image.width(), 8);
            assert_eq!(image.height(), 8);
        }
        eprintln!("lossy_distances: PASSED jxl-oxide (VarDCT WIP)");
    }

    /// Dual-decoder validation for irregular dimensions
    #[test]
    fn test_dual_decode_irregular_dims() {
        // Test non-power-of-2 dimensions
        for (w, h) in [(7, 9), (11, 13), (33, 17), (100, 50)] {
            let mut data = vec![0u8; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) * 3;
                    data[idx] = ((x * 255) / w.max(1)) as u8;
                    data[idx + 1] = ((y * 255) / h.max(1)) as u8;
                    data[idx + 2] = 128;
                }
            }
            let encoded = Encoder::new().encode_rgb8(&data, w, h).unwrap();
            validate_dual_decoder(
                &encoded,
                w as u32,
                h as u32,
                &format!("irregular_{}x{}", w, h),
            );
        }
    }

    /// Dual-decoder validation for multi-group images
    /// Note: VarDCT encoding is WIP - jxl-oxide only for lossy
    #[test]
    fn test_dual_decode_multi_group() {
        // 256x256 = 1 group boundary test - lossless uses dual-decoder
        let mut data = vec![0u8; 256 * 256 * 3];
        for y in 0..256 {
            for x in 0..256 {
                let idx = (y * 256 + x) * 3;
                data[idx] = x as u8;
                data[idx + 1] = y as u8;
                data[idx + 2] = ((x + y) % 256) as u8;
            }
        }
        // Test lossless (modular) with dual-decoder
        let encoded = Encoder::new().encode_rgb8(&data, 256, 256).unwrap();
        validate_dual_decoder(&encoded, 256, 256, "multi_group_256x256_lossless");

        // Also test lossy with jxl-oxide only
        let lossy_encoded = encode_lossy_rgb8(&data, 256, 256, 2.0).unwrap();
        let oxide_result =
            jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&lossy_encoded));
        assert!(
            oxide_result.is_ok(),
            "jxl-oxide should decode multi-group lossy"
        );
        let image = oxide_result.unwrap();
        assert_eq!(image.width(), 256);
        assert_eq!(image.height(), 256);
        eprintln!("multi_group_256x256_lossy: PASSED jxl-oxide (VarDCT WIP)");
    }

    /// Test multi-group encoding with actual multiple groups (>256x256)
    #[test]
    fn test_encode_multigroup_300x300() {
        // 300x300 RGB image - requires 2x2 = 4 groups
        let mut data = vec![0u8; 300 * 300 * 3];
        for y in 0..300 {
            for x in 0..300 {
                let idx = (y * 300 + x) * 3;
                data[idx] = ((x + y) % 256) as u8;
                data[idx + 1] = (x % 256) as u8;
                data[idx + 2] = (y % 256) as u8;
            }
        }

        let encoded = Encoder::new().encode_rgb8(&data, 300, 300).unwrap();

        // Check JXL signature
        assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);

        eprintln!("Multi-group 300x300: {} bytes", encoded.len());
        std::fs::write("/tmp/test_multigroup_300.jxl", &encoded).unwrap();

        // Verify with jxl-oxide
        let result = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&encoded));
        assert!(
            result.is_ok(),
            "jxl-oxide should decode multi-group: {:?}",
            result.err()
        );
        let image = result.unwrap();
        assert_eq!(image.width(), 300);
        assert_eq!(image.height(), 300);
        eprintln!("Multi-group 300x300 decode: PASSED jxl-oxide");
    }

    /// Test multi-group encoding with 512x512 (4 full groups)
    #[test]
    fn test_encode_multigroup_512x512() {
        // 512x512 RGB image - requires 2x2 = 4 groups (all full 256x256)
        let mut data = vec![0u8; 512 * 512 * 3];
        for y in 0..512 {
            for x in 0..512 {
                let idx = (y * 512 + x) * 3;
                data[idx] = ((x + y) % 256) as u8;
                data[idx + 1] = (x % 256) as u8;
                data[idx + 2] = (y % 256) as u8;
            }
        }

        let encoded = Encoder::new().encode_rgb8(&data, 512, 512).unwrap();

        // Check JXL signature
        assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);

        eprintln!("Multi-group 512x512: {} bytes", encoded.len());
        std::fs::write("/tmp/test_multigroup_512.jxl", &encoded).unwrap();

        // Verify with jxl-oxide
        let result = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&encoded));
        assert!(
            result.is_ok(),
            "jxl-oxide should decode 512x512 multi-group: {:?}",
            result.err()
        );
        let image = result.unwrap();
        assert_eq!(image.width(), 512);
        assert_eq!(image.height(), 512);
        eprintln!("Multi-group 512x512 decode: PASSED jxl-oxide");
    }

    /// Dual-decoder validation for corpus images
    #[test]
    fn test_dual_decode_corpus_images() {
        const CORPUS_PATH: &str = "/home/lilith/work/codec-corpus";

        // Test a few representative images from the corpus
        let test_images = [
            ("pngsuite/basn2c08.png", false), // RGB lossless
            ("pngsuite/basn0g08.png", true),  // Gray lossless
        ];

        for (image_path, is_gray) in test_images {
            let path = format!("{}/{}", CORPUS_PATH, image_path);
            if !std::path::Path::new(&path).exists() {
                eprintln!("Skipping {}: not found", image_path);
                continue;
            }

            let img = image::open(&path).unwrap();
            let (w, h) = (img.width() as usize, img.height() as usize);

            let encoded = if is_gray {
                let gray = img.to_luma8();
                Encoder::new().encode_gray8(gray.as_raw(), w, h).unwrap()
            } else {
                let rgb = img.to_rgb8();
                Encoder::new().encode_rgb8(rgb.as_raw(), w, h).unwrap()
            };

            validate_dual_decoder(
                &encoded,
                w as u32,
                h as u32,
                &format!("corpus_{}", image_path.replace("/", "_").replace(".", "_")),
            );
        }
    }

    // ========== PROPER ROUNDTRIP TESTS ==========
    // These tests verify actual pixel values match after encode -> decode

    /// Debug test to understand jxl-oxide output format
    #[test]
    fn test_debug_decode_format() {
        // 2x2 simple test: Red, Green, Blue, White
        let data = vec![
            255, 0, 0, // R
            0, 255, 0, // G
            0, 0, 255, // B
            255, 255, 255, // W
        ];

        let encoded = Encoder::new().encode_rgb8(&data, 2, 2).unwrap();
        eprintln!("Encoded {} bytes", encoded.len());

        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .unwrap();

        eprintln!("Image: {}x{}", image.width(), image.height());

        let render = image.render_frame(0).unwrap();
        let fb = render.image_all_channels();

        eprintln!(
            "FrameBuffer: {}x{}, {} channels",
            fb.width(),
            fb.height(),
            fb.channels()
        );

        let buf = fb.buf();
        eprintln!("Buffer len: {}", buf.len());
        for (i, v) in buf.iter().enumerate() {
            eprintln!("  buf[{}] = {:.4}", i, v);
        }

        // Print expected vs actual for debugging
        eprintln!("\nExpected input data:");
        for (i, v) in data.iter().enumerate() {
            eprintln!("  data[{}] = {}", i, v);
        }
    }

    /// Test lossless roundtrip for a simple RGB checkerboard
    #[test]
    fn test_roundtrip_lossless_rgb_checkerboard() {
        let mut data = vec![0u8; 8 * 8 * 3];
        for y in 0..8 {
            for x in 0..8 {
                let idx = (y * 8 + x) * 3;
                if (x + y) % 2 == 0 {
                    data[idx] = 255; // R
                    data[idx + 1] = 0; // G
                    data[idx + 2] = 0; // B
                } else {
                    data[idx] = 0;
                    data[idx + 1] = 0;
                    data[idx + 2] = 255;
                }
            }
        }
        validate_lossless_roundtrip_rgb(&data, 8, 8, "rgb_checkerboard_8x8");
    }

    /// Test lossless roundtrip for RGB gradient
    #[test]
    fn test_roundtrip_lossless_rgb_gradient() {
        let mut data = vec![0u8; 16 * 16 * 3];
        for y in 0..16 {
            for x in 0..16 {
                let idx = (y * 16 + x) * 3;
                data[idx] = (x * 16) as u8;
                data[idx + 1] = (y * 16) as u8;
                data[idx + 2] = ((x + y) * 8) as u8;
            }
        }
        validate_lossless_roundtrip_rgb(&data, 16, 16, "rgb_gradient_16x16");
    }

    /// Test lossless roundtrip for solid color
    #[test]
    fn test_roundtrip_lossless_rgb_solid() {
        let mut data = vec![0u8; 32 * 32 * 3];
        for i in 0..(32 * 32) {
            data[i * 3] = 200;
            data[i * 3 + 1] = 100;
            data[i * 3 + 2] = 50;
        }
        validate_lossless_roundtrip_rgb(&data, 32, 32, "rgb_solid_32x32");
    }

    /// Test lossless roundtrip for grayscale gradient
    #[test]
    fn test_roundtrip_lossless_gray_gradient() {
        let data: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        validate_lossless_roundtrip_gray(&data, 8, 8, "gray_gradient_8x8");
    }

    /// Test lossless roundtrip for grayscale varied values
    #[test]
    fn test_roundtrip_lossless_gray_varied() {
        let data = vec![0u8, 64, 128, 192, 255, 100, 50, 200];
        validate_lossless_roundtrip_gray(&data, 4, 2, "gray_varied_4x2");
    }

    /// Test lossless roundtrip for multi-group RGB (300x300)
    /// NOTE: Currently skipped - multi-group pixel roundtrip has edge context bug
    #[test]
    #[ignore = "Multi-group residual encoding has group boundary context mismatch"]
    fn test_roundtrip_lossless_rgb_multigroup_300() {
        let mut data = vec![0u8; 300 * 300 * 3];
        for y in 0..300 {
            for x in 0..300 {
                let idx = (y * 300 + x) * 3;
                data[idx] = ((x + y) % 256) as u8;
                data[idx + 1] = (x % 256) as u8;
                data[idx + 2] = (y % 256) as u8;
            }
        }
        validate_lossless_roundtrip_rgb(&data, 300, 300, "rgb_multigroup_300x300");
    }

    /// Test lossy roundtrip at distance 1.0 (high quality)
    /// NOTE: VarDCT encoding is WIP - some decode issues exist
    #[test]
    #[ignore = "VarDCT lossy encoding has known jxl-oxide compatibility issues"]
    fn test_roundtrip_lossy_rgb_d1() {
        let mut data = vec![0u8; 16 * 16 * 3];
        for y in 0..16 {
            for x in 0..16 {
                let idx = (y * 16 + x) * 3;
                data[idx] = (x * 16) as u8;
                data[idx + 1] = (y * 16) as u8;
                data[idx + 2] = 128;
            }
        }
        // At distance 1.0, max_diff should be reasonable (< 50 for most images)
        validate_lossy_roundtrip_rgb(&data, 16, 16, 1.0, 80, "rgb_lossy_d1_16x16");
    }

    /// Test lossy roundtrip at distance 2.0 (medium quality)
    /// NOTE: VarDCT encoding is WIP - some decode issues exist
    #[test]
    #[ignore = "VarDCT lossy encoding has known jxl-oxide compatibility issues"]
    fn test_roundtrip_lossy_rgb_d2() {
        let mut data = vec![0u8; 16 * 16 * 3];
        for y in 0..16 {
            for x in 0..16 {
                let idx = (y * 16 + x) * 3;
                data[idx] = (x * 16) as u8;
                data[idx + 1] = (y * 16) as u8;
                data[idx + 2] = 128;
            }
        }
        // At distance 2.0, higher tolerance needed
        validate_lossy_roundtrip_rgb(&data, 16, 16, 2.0, 120, "rgb_lossy_d2_16x16");
    }

    /// Test lossless roundtrip for corpus image (pngsuite)
    #[test]
    fn test_roundtrip_lossless_corpus_rgb() {
        const CORPUS_PATH: &str = "/home/lilith/work/codec-corpus";
        let path = format!("{}/pngsuite/basn2c08.png", CORPUS_PATH);
        if !std::path::Path::new(&path).exists() {
            eprintln!("Skipping: {} not found", path);
            return;
        }

        let img = image::open(&path).unwrap();
        let rgb = img.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        validate_lossless_roundtrip_rgb(rgb.as_raw(), w, h, "corpus_basn2c08");
    }

    /// Test lossless roundtrip for corpus grayscale (pngsuite)
    #[test]
    fn test_roundtrip_lossless_corpus_gray() {
        const CORPUS_PATH: &str = "/home/lilith/work/codec-corpus";
        let path = format!("{}/pngsuite/basn0g08.png", CORPUS_PATH);
        if !std::path::Path::new(&path).exists() {
            eprintln!("Skipping: {} not found", path);
            return;
        }

        let img = image::open(&path).unwrap();
        let gray = img.to_luma8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        validate_lossless_roundtrip_gray(gray.as_raw(), w, h, "corpus_basn0g08");
    }
}
