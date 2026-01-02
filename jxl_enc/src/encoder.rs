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
        std::fs::write("/tmp/lossy_8x8.jxl", &encoded).unwrap();
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
}
