// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Tests for the encoder — uses the public LosslessConfig/LossyConfig API.

use crate::{LosslessConfig, LossyConfig, PixelLayout, ProgressiveMode};

mod tests {
    use crate::{LosslessConfig, LossyConfig, PixelLayout};

    #[test]
    fn test_encode_small_rgb() {
        // 2x2 RGB image
        let data = vec![
            255, 0, 0, // Red
            0, 255, 0, // Green
            0, 0, 255, // Blue
            255, 255, 0, // Yellow
        ];

        let encoded = LosslessConfig::new()
            .encode(&data, 2, 2, PixelLayout::Rgb8)
            .unwrap();

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

        // Write to temp dir for debugging
        std::fs::write(std::env::temp_dir().join("test_out.jxl"), &encoded).unwrap();
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

        let encoded = LosslessConfig::new()
            .encode(&data, 4, 4, PixelLayout::Rgb8)
            .unwrap();
        // Check JXL signature
        assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);
    }

    #[test]
    fn test_encode_flat() {
        // 4x4 flat gray image
        let data = vec![128u8; 4 * 4 * 3];

        let encoded = LosslessConfig::new()
            .encode(&data, 4, 4, PixelLayout::Rgb8)
            .unwrap();
        // Check JXL signature
        assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);

        // Flat images should compress very well
        // (though our minimal encoder may not be optimal)
    }

    #[test]
    fn test_encode_black_2x2() {
        // 2x2 all-black image - should work with minimal zero-everywhere encoding
        let data = vec![0u8; 2 * 2 * 3]; // All zeros

        let encoded = LosslessConfig::new()
            .encode(&data, 2, 2, PixelLayout::Rgb8)
            .unwrap();

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

        // Write to temp dir for testing with decoder
        std::fs::write(std::env::temp_dir().join("test_black.jxl"), &encoded).unwrap();
    }

    #[test]
    fn test_encode_white_2x2() {
        // 2x2 all-white image - tests single non-zero symbol encoding
        let data = vec![255u8; 2 * 2 * 3]; // All 255

        let encoded = LosslessConfig::new()
            .encode(&data, 2, 2, PixelLayout::Rgb8)
            .unwrap();
        assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);

        eprintln!("Encoded white 2x2: {} bytes:", encoded.len());
        for (i, b) in encoded.iter().enumerate() {
            eprint!("{:02x} ", b);
            if (i + 1) % 16 == 0 {
                eprintln!();
            }
        }
        eprintln!();

        std::fs::write(std::env::temp_dir().join("test_white.jxl"), &encoded).unwrap();
    }

    #[test]
    fn test_config_defaults() {
        let cfg = LosslessConfig::new();
        assert_eq!(cfg.effort(), 7);
        assert!(cfg.ans());

        let cfg = LossyConfig::new(1.0);
        assert_eq!(cfg.distance(), 1.0);
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

        let encoded = LossyConfig::new(1.0)
            .encode(&data, 8, 8, PixelLayout::Rgb8)
            .unwrap();

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
        let lossy_8x8_path = std::env::temp_dir().join("lossy_8x8.jxl");
        std::fs::write(&lossy_8x8_path, &encoded).unwrap();

        // Verify roundtrip: file bytes == memory bytes
        let read_back = std::fs::read(&lossy_8x8_path).unwrap();
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

        // Try to actually render the frame (not just parse headers)
        let _render = image
            .render_frame(0)
            .expect("test_encode_lossy_8x8: render failed");

        eprintln!(
            "Decode from file bytes succeeded: {}x{}, rendered successfully",
            image.width(),
            image.height()
        );
    }
}

mod gray_tests {
    use crate::{LosslessConfig, PixelLayout};

    #[test]
    fn test_encode_gray_2x2() {
        // 2x2 grayscale with varied values
        let data = vec![0u8, 128, 64, 255];

        let encoded = LosslessConfig::new()
            .encode(&data, 2, 2, PixelLayout::Gray8)
            .unwrap();

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

        std::fs::write(std::env::temp_dir().join("test_gray.jxl"), &encoded).unwrap();
    }
}

#[test]
fn test_encode_gray_binary() {
    // 2x2 grayscale with only 0 and 255
    let data = vec![0u8, 255, 0, 255];

    let encoded = LosslessConfig::new()
        .encode(&data, 2, 2, PixelLayout::Gray8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("test_gray_binary.jxl"), &encoded).unwrap();

    eprintln!("Encoded gray binary 2x2: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_uniform_128() {
    // 2x2 grayscale all 128
    let data = vec![128u8; 4];

    let encoded = LosslessConfig::new()
        .encode(&data, 2, 2, PixelLayout::Gray8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("test_gray_128.jxl"), &encoded).unwrap();
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

    let encoded = LosslessConfig::new()
        .encode(&data, 2, 2, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("test_rgb_gray.jxl"), &encoded).unwrap();
    eprintln!("Encoded RGB simulated gray: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_0_and_1() {
    // 2x2 grayscale with only 0 and 1
    let data = vec![0u8, 1, 0, 1];

    let encoded = LosslessConfig::new()
        .encode(&data, 2, 2, PixelLayout::Gray8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("test_gray_01.jxl"), &encoded).unwrap();
    eprintln!("Encoded gray 0/1 2x2: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_0_and_3() {
    // 2x2 grayscale with 0 and 3 (zigzag: 0, 6)
    let data = vec![0u8, 3, 0, 3];

    let encoded = LosslessConfig::new()
        .encode(&data, 2, 2, PixelLayout::Gray8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("test_gray_03.jxl"), &encoded).unwrap();
    eprintln!("Encoded gray 0/3 2x2: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_0_and_7() {
    // 2x2 grayscale with 0 and 7 (zigzag: 0, 14)
    let data = vec![0u8, 7, 0, 7];

    let encoded = LosslessConfig::new()
        .encode(&data, 2, 2, PixelLayout::Gray8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("test_gray_07.jxl"), &encoded).unwrap();
    eprintln!("Encoded gray 0/7 2x2: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_0_and_15() {
    // 2x2 grayscale with 0 and 15 (zigzag: 0, 30)
    let data = vec![0u8, 15, 0, 15];

    let encoded = LosslessConfig::new()
        .encode(&data, 2, 2, PixelLayout::Gray8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("test_gray_015.jxl"), &encoded).unwrap();
    eprintln!("Encoded gray 0/15 2x2: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_0_and_4() {
    // 2x2 grayscale with 0 and 4 (zigzag: 0, 8) - boundary: al_size=9, max_bits=4
    let data = vec![0u8, 4, 0, 4];

    let encoded = LosslessConfig::new()
        .encode(&data, 2, 2, PixelLayout::Gray8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("test_gray_04.jxl"), &encoded).unwrap();
    eprintln!("Encoded gray 0/4 2x2: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_1_and_2() {
    // 2x2 grayscale with 1 and 2 (zigzag: 2, 4) - al_size=5, max_bits=3
    let data = vec![1u8, 2, 1, 2];

    let encoded = LosslessConfig::new()
        .encode(&data, 2, 2, PixelLayout::Gray8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("test_gray_12.jxl"), &encoded).unwrap();
    eprintln!("Encoded gray 1/2 2x2: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_4x4_pattern() {
    // 4x4 grayscale with 4 unique values (max for simple Huffman)
    let data: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];

    let encoded = LosslessConfig::new()
        .encode(&data, 4, 4, PixelLayout::Gray8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("test_gray_4x4.jxl"), &encoded).unwrap();
    eprintln!("Encoded gray 4x4 pattern: {} bytes", encoded.len());
}

#[test]
fn test_encode_gray_16_symbols() {
    // 4x4 gradient with 16 unique values - now works with full Huffman
    let data: Vec<u8> = (0u8..16).collect();

    let encoded = LosslessConfig::new()
        .encode(&data, 4, 4, PixelLayout::Gray8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("test_gray_16sym.jxl"), &encoded).unwrap();
    eprintln!("Encoded 16-symbol gray 4x4: {} bytes", encoded.len());

    // Check JXL signature
    assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);
}

#[test]
fn test_encode_gray_256_symbols() {
    // 16x16 gradient with 256 unique values
    let data: Vec<u8> = (0u8..=255).collect();

    let encoded = LosslessConfig::new()
        .encode(&data, 16, 16, PixelLayout::Gray8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("test_gray_256sym.jxl"), &encoded).unwrap();
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

    let encoded = LosslessConfig::new()
        .encode(&data, 8, 8, PixelLayout::Gray8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("test_gray_8x8.jxl"), &encoded).unwrap();
    eprintln!("Encoded gray 8x8 checkerboard: {} bytes", encoded.len());
}

mod corpus_tests {
    use crate::{LosslessConfig, PixelLayout};

    fn corpus_path_string() -> String {
        crate::test_helpers::corpus_dir()
            .to_string_lossy()
            .into_owned()
    }

    fn test_image_roundtrip(path: &str) -> Result<(usize, usize, usize), String> {
        let img = image::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;

        let (width, height) = (img.width() as usize, img.height() as usize);

        // Convert to appropriate format and encode
        let encoded = match img.color() {
            image::ColorType::L8 => {
                let gray = img.to_luma8();
                LosslessConfig::new()
                    .encode(
                        gray.as_raw(),
                        width as u32,
                        height as u32,
                        PixelLayout::Gray8,
                    )
                    .map_err(|e| format!("Encode failed: {}", e))?
            }
            image::ColorType::Rgb8 => {
                let rgb = img.to_rgb8();
                LosslessConfig::new()
                    .encode(rgb.as_raw(), width as u32, height as u32, PixelLayout::Rgb8)
                    .map_err(|e| format!("Encode failed: {}", e))?
            }
            image::ColorType::Rgba8 => {
                let rgba = img.to_rgba8();
                LosslessConfig::new()
                    .encode(
                        rgba.as_raw(),
                        width as u32,
                        height as u32,
                        PixelLayout::Rgba8,
                    )
                    .map_err(|e| format!("Encode failed: {}", e))?
            }
            other => {
                // Convert to RGB8 for other formats
                let rgb = img.to_rgb8();
                LosslessConfig::new()
                    .encode(rgb.as_raw(), width as u32, height as u32, PixelLayout::Rgb8)
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
        crate::skip_without_corpus!();
        // 8-bit grayscale from PNG suite
        let path = format!("{}/pngsuite/basi0g08.png", corpus_path_string());
        if std::path::Path::new(&path).exists() {
            let img = image::open(&path).unwrap();
            let gray = img.to_luma8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let encoded = LosslessConfig::new()
                .encode(gray.as_raw(), w as u32, h as u32, PixelLayout::Gray8)
                .unwrap();
            std::fs::write(std::env::temp_dir().join("pngsuite_gray.jxl"), &encoded).unwrap();
            eprintln!("basi0g08.png: {}x{} -> {} bytes", w, h, encoded.len());
        } else {
            eprintln!("Skipping: {} not found", path);
        }
    }

    #[test]
    fn test_pngsuite_rgb() {
        crate::skip_without_corpus!();
        // 8-bit RGB from PNG suite
        let path = format!(
            "{}/pngsuite/basi2c08.png",
            crate::test_helpers::corpus_dir().display()
        );
        if std::path::Path::new(&path).exists() {
            let img = image::open(&path).unwrap();
            let rgb = img.to_rgb8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let encoded = LosslessConfig::new()
                .encode(rgb.as_raw(), w as u32, h as u32, PixelLayout::Rgb8)
                .unwrap();
            std::fs::write(std::env::temp_dir().join("pngsuite_rgb.jxl"), &encoded).unwrap();
            eprintln!("basi2c08.png: {}x{} -> {} bytes", w, h, encoded.len());
        } else {
            eprintln!("Skipping: {} not found", path);
        }
    }

    #[test]
    fn test_kodak_01() {
        crate::skip_without_corpus!();
        let path = format!(
            "{}/kodak/1.png",
            crate::test_helpers::corpus_dir().display()
        );
        if std::path::Path::new(&path).exists() {
            match test_image_roundtrip(&path) {
                Ok((w, h, size)) => {
                    eprintln!("kodak/1.png: {}x{} -> {} bytes", w, h, size);
                    // Save for manual verification
                    let img = image::open(&path).unwrap();
                    let rgb = img.to_rgb8();
                    let encoded = LosslessConfig::new()
                        .encode(rgb.as_raw(), w as u32, h as u32, PixelLayout::Rgb8)
                        .unwrap();
                    std::fs::write(std::env::temp_dir().join("kodak1.jxl"), &encoded).unwrap();
                }
                Err(e) => panic!("{}", e),
            }
        } else {
            eprintln!("Skipping: {} not found", path);
        }
    }

    #[test]
    fn test_corpus_batch() {
        crate::skip_without_corpus!();
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
            let full_path = format!(
                "{}/{}",
                crate::test_helpers::corpus_dir().display(),
                img_path
            );
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

    let encoded = LosslessConfig::new()
        .encode(&data, 8, 8, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("rgb_8x8.jxl"), &encoded).unwrap();
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

    let encoded = LosslessConfig::new()
        .encode(&data, 8, 8, PixelLayout::Gray8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("gray_8x8.jxl"), &encoded).unwrap();
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

    let encoded = LosslessConfig::new()
        .encode(&data, 4, 4, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("rgb_4x4.jxl"), &encoded).unwrap();
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
    let encoded = LosslessConfig::new()
        .encode(&data, 6, 6, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("rgb_6x6.jxl"), &encoded).unwrap();
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
    let encoded = LosslessConfig::new()
        .encode(&data, 7, 7, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("rgb_7x7.jxl"), &encoded).unwrap();
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
    let encoded = LosslessConfig::new()
        .encode(&data, 9, 9, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("rgb_9x9.jxl"), &encoded).unwrap();
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
    let encoded = LosslessConfig::new()
        .encode(&data, 16, 16, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("rgb_16x16.jxl"), &encoded).unwrap();
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
    let encoded = LosslessConfig::new()
        .encode(&data, 24, 24, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("rgb_24x24.jxl"), &encoded).unwrap();
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
    let encoded = LosslessConfig::new()
        .encode(&data, 10, 10, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("rgb_10x10.jxl"), &encoded).unwrap();
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
    let encoded = LosslessConfig::new()
        .encode(&data, 32, 32, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("rgb_32x32.jxl"), &encoded).unwrap();
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
    let encoded = LosslessConfig::new()
        .encode(&data, 64, 64, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("rgb_64x64.jxl"), &encoded).unwrap();
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
    let encoded = LosslessConfig::new()
        .encode(&data, 256, 256, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(std::env::temp_dir().join("rgb_256x256.jxl"), &encoded).unwrap();
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
        let encoded = LosslessConfig::new()
            .encode(&data, w as u32, h as u32, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("{}x{} failed to encode: {}", w, h, e));

        let path = std::env::temp_dir().join(format!("rgb_{}x{}.jxl", w, h));
        let _ = std::fs::write(&path, &encoded);
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
    let encoded = LosslessConfig::new()
        .encode(&data, 32, 32, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(
        std::env::temp_dir().join("rgb_gradient_32x32.jxl"),
        &encoded,
    )
    .unwrap();
    eprintln!("Encoded RGB gradient 32x32: {} bytes", encoded.len());
}

mod decoder_validation {
    use crate::{LosslessConfig, LossyConfig, PixelLayout};
    use std::process::Command;

    fn djxl_path_string() -> String {
        crate::test_helpers::djxl_path()
    }

    /// Validates lossless roundtrip: encode -> decode -> compare pixels exactly.
    ///
    /// Uses jxl-rs as primary decoder (per CLAUDE.md). Falls back to jxl-oxide
    /// for single-group images where both work. jxl-oxide has a known limitation
    /// with ANS entropy coding in multi-group modular frames.
    ///
    /// Returns the decoded pixel data on success.
    fn validate_lossless_roundtrip_rgb(
        original: &[u8],
        width: usize,
        height: usize,
        test_name: &str,
    ) -> Vec<u8> {
        validate_lossless_roundtrip_rgb_config(
            original,
            width,
            height,
            test_name,
            LosslessConfig::new(),
        )
    }

    fn validate_lossless_roundtrip_rgb_config(
        original: &[u8],
        width: usize,
        height: usize,
        test_name: &str,
        config: LosslessConfig,
    ) -> Vec<u8> {
        assert_eq!(original.len(), width * height * 3);

        // Encode
        let encoded = config
            .encode(original, width as u32, height as u32, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("{}: encoding failed: {}", test_name, e));

        // Save to file for debugging
        let path = std::env::temp_dir().join(format!("{}.jxl", test_name));
        let _ = std::fs::write(&path, &encoded);
        eprintln!(
            "{}: Saved {} bytes to {}",
            test_name,
            encoded.len(),
            path.display()
        );

        // Decode with jxl-rs (PRIMARY decoder)
        let jxlrs_result = crate::test_helpers::decode_with_jxl_rs(&encoded);
        let decoded = match jxlrs_result {
            Ok(decoded_img) => {
                assert_eq!(decoded_img.width, width);
                assert_eq!(decoded_img.height, height);

                // Convert f32 to u8
                let decoded: Vec<u8> = decoded_img
                    .pixels
                    .iter()
                    .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
                    .collect();
                eprintln!("{}: jxl-rs decode OK", test_name);
                decoded
            }
            Err(e) => {
                panic!("{}: jxl-rs decode failed: {}", test_name, e);
            }
        };

        // Also verify with jxl-oxide (secondary decoder, may fail for multi-group ANS)
        match jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&encoded)) {
            Ok(image) => match image.render_frame(0) {
                Ok(_render) => {
                    eprintln!("{}: jxl-oxide decode OK (secondary)", test_name);
                }
                Err(e) => {
                    eprintln!(
                        "{}: jxl-oxide render failed (non-fatal, jxl-rs succeeded): {}",
                        test_name, e
                    );
                }
            },
            Err(e) => {
                eprintln!(
                    "{}: jxl-oxide read failed (non-fatal, jxl-rs succeeded): {}",
                    test_name, e
                );
            }
        }

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
        let encoded = LosslessConfig::new()
            .encode(original, width as u32, height as u32, PixelLayout::Gray8)
            .unwrap_or_else(|e| panic!("{}: encoding failed: {}", test_name, e));

        // Decode with jxl-oxide
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .unwrap_or_else(|e| panic!("{}: jxl-oxide decode failed: {}", test_name, e));

        assert_eq!(image.width() as usize, width);
        assert_eq!(image.height() as usize, height);

        // Render frame and extract pixels
        let render = image
            .render_frame(0)
            .unwrap_or_else(|e| panic!("{}: render failed: {}", test_name, e));

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
        for (&orig, &dec) in original.iter().zip(decoded.iter()) {
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
        let encoded = LossyConfig::new(distance)
            .encode(original, width as u32, height as u32, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("{}: encoding failed: {}", test_name, e));

        // Decode with jxl-oxide
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .unwrap_or_else(|e| panic!("{}: jxl-oxide decode failed: {}", test_name, e));

        assert_eq!(image.width() as usize, width);
        assert_eq!(image.height() as usize, height);

        // Render frame and extract pixels
        let render = image
            .render_frame(0)
            .unwrap_or_else(|e| panic!("{}: render failed: {}", test_name, e));

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
        let djxl = djxl_path_string();
        if std::path::Path::new(&djxl).exists() {
            // Write JXL to temp file
            let temp_jxl = std::env::temp_dir().join(format!(
                "dual_decode_test_{}.jxl",
                test_name.replace(" ", "_")
            ));
            let temp_png = std::env::temp_dir().join(format!(
                "dual_decode_test_{}.png",
                test_name.replace(" ", "_")
            ));
            std::fs::write(&temp_jxl, encoded).expect("Failed to write temp JXL");

            // Run djxl — may fail if binary exists but shared libs are missing
            // (e.g. cross-compilation container with host volume mount)
            match Command::new(&djxl).args([&temp_jxl, &temp_png]).output() {
                Ok(output) if output.status.success() => {
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
                    eprintln!(
                        "{}: PASSED dual-decoder validation (jxl-oxide + djxl)",
                        test_name
                    );
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    if stderr.contains("cannot open shared object")
                        || stderr.contains("No such file or directory")
                    {
                        eprintln!(
                            "{}: PASSED jxl-oxide only (djxl missing shared libs)",
                            test_name
                        );
                    } else {
                        panic!("{}: djxl decode failed: {}", test_name, stderr);
                    }
                }
                Err(e) => {
                    eprintln!(
                        "{}: PASSED jxl-oxide only (djxl not runnable: {})",
                        test_name, e
                    );
                }
            }

            // Cleanup
            let _ = std::fs::remove_file(&temp_jxl);
            let _ = std::fs::remove_file(&temp_png);
        } else {
            eprintln!(
                "{}: PASSED jxl-oxide only (djxl not available at {})",
                test_name, djxl
            );
        }

        oxide_dims
    }

    /// Test that our encoded files can be decoded by jxl-oxide
    #[test]
    fn test_decode_simple_gray() {
        // 2x2 grayscale with values [0, 1, 0, 1]
        let data = vec![0u8, 1, 0, 1];
        let encoded = LosslessConfig::new()
            .encode(&data, 2, 2, PixelLayout::Gray8)
            .unwrap();

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

        let encoded = LosslessConfig::new()
            .encode(&data, 8, 8, PixelLayout::Rgb8)
            .unwrap();
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

        let encoded = LossyConfig::new(1.0)
            .encode(&data, 8, 8, PixelLayout::Rgb8)
            .unwrap();
        eprintln!("Lossy 8x8 encoded to {} bytes", encoded.len());

        // Try to decode with jxl-oxide
        let decoder = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&encoded));

        match decoder {
            Ok(image) => {
                assert_eq!(image.width(), 8);
                assert_eq!(image.height(), 8);

                // Try to actually render the frame (not just parse headers)
                let _render = image
                    .render_frame(0)
                    .expect("test_decode_lossy_rgb: render failed");

                eprintln!(
                    "Successfully decoded lossy 8x8: {}x{}, rendered successfully",
                    image.width(),
                    image.height()
                );
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

        let encoded = LossyConfig::new(1.0)
            .encode(&data, 8, 8, PixelLayout::Rgb8)
            .unwrap();
        eprintln!("Solid color 8x8 encoded to {} bytes", encoded.len());

        // Decode with jxl-oxide
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .expect("Failed to decode");

        assert_eq!(image.width(), 8);
        assert_eq!(image.height(), 8);

        // Try to actually render the frame (not just parse headers)
        let _render = image
            .render_frame(0)
            .expect("test_decode_lossy_solid_color: render failed");

        eprintln!(
            "Successfully decoded solid color 8x8: {}x{}, rendered successfully",
            image.width(),
            image.height()
        );
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
            let encoded = LossyConfig::new(distance)
                .encode(&data, 8, 8, PixelLayout::Rgb8)
                .unwrap();
            eprintln!("Distance {}: {} bytes", distance, encoded.len());

            // Verify decodes correctly
            let image = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&encoded))
                .unwrap_or_else(|e| panic!("Failed to decode at distance {}: {}", distance, e));

            assert_eq!(image.width(), 8);
            assert_eq!(image.height(), 8);

            // Try to actually render the frame (not just parse headers)
            let _render = image.render_frame(0).unwrap_or_else(|e| {
                panic!(
                    "test_decode_lossy_distances: render failed at distance {}: {}",
                    distance, e
                )
            });
        }
    }

    /// Test that pngsuite RGB images can be decoded by jxl-oxide
    #[test]
    fn test_decode_pngsuite_rgb() {
        crate::skip_without_corpus!();
        // (corpus path resolved via crate::test_helpers::corpus_dir)
        let path = format!(
            "{}/pngsuite/basn2c08.png",
            crate::test_helpers::corpus_dir().display()
        );
        if !std::path::Path::new(&path).exists() {
            eprintln!("Skipping test: {} not found", path);
            return;
        }

        let img = image::open(&path).unwrap();
        let rgb = img.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);

        let encoded = LosslessConfig::new()
            .encode(rgb.as_raw(), w as u32, h as u32, PixelLayout::Rgb8)
            .unwrap();
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

        let encoded = LossyConfig::new(2.0)
            .encode(&data, 512, 512, PixelLayout::Rgb8)
            .unwrap();
        eprintln!("Multi-group 512x512 encoded to {} bytes", encoded.len());

        // Verify decodes correctly with jxl-oxide
        match jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&encoded)) {
            Ok(image) => {
                assert_eq!(image.width(), 512);
                assert_eq!(image.height(), 512);

                // Try to actually render the frame (not just parse headers)
                let _render = image
                    .render_frame(0)
                    .expect("test_decode_lossy_multi_group: render failed");

                eprintln!(
                    "Successfully decoded 512x512: {}x{}, rendered successfully",
                    image.width(),
                    image.height()
                );
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

    /// Test VarDCT lossy encoding for images requiring multiple DC groups (>2048px).
    /// This is a regression test for imazen/jxl-encoder#3 where a context tree
    /// mismatch caused decode failures on wide images.
    #[test]
    fn test_lossy_multi_dc_group_roundtrip() {
        // 2100x256: requires 2 DC groups in x (ceil(2100/2048) = 2)
        let (w, h) = (2100, 256);
        let mut data = vec![0u8; w * h * 3];
        let mut seed = 42u64;
        for val in data.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *val = (seed >> 56) as u8;
        }

        for effort in [3, 5, 7] {
            let encoded = LossyConfig::new(2.0)
                .with_effort(effort)
                .encode(&data, w as u32, h as u32, PixelLayout::Rgb8)
                .unwrap();

            // Verify with jxl-oxide
            let image = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&encoded))
                .unwrap_or_else(|e| panic!("jxl-oxide decode failed for {w}x{h} e{effort}: {e:?}"));
            assert_eq!(image.width(), w as u32);
            assert_eq!(image.height(), h as u32);
            let _render = image
                .render_frame(0)
                .unwrap_or_else(|e| panic!("jxl-oxide render failed for {w}x{h} e{effort}: {e:?}"));

            // Verify with djxl
            let djxl = djxl_path_string();
            if std::path::Path::new(&djxl).exists() {
                let tmp = std::env::temp_dir().join(format!("multi_dc_{w}x{h}_e{effort}.jxl"));
                std::fs::write(&tmp, &encoded).unwrap();
                let out = std::env::temp_dir().join(format!("multi_dc_{w}x{h}_e{effort}_dec.png"));
                let result = Command::new(&djxl)
                    .args([&tmp, &out])
                    .output()
                    .expect("djxl failed to run");
                assert!(
                    result.status.success(),
                    "djxl failed for {w}x{h} e{effort}: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
            }
        }
    }

    /// Test multi-DC-group with varying widths to cover different DC group counts.
    #[test]
    fn test_lossy_multi_dc_group_widths() {
        // Each width exercises a different number of DC groups:
        // 2049 → 2 DC groups, 4097 → 3, 6145 → 4
        for w in [2049u32, 4097, 6145] {
            let h = 64u32;
            let mut data = vec![128u8; (w * h * 3) as usize];
            let mut seed = w as u64;
            for val in data.iter_mut() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                *val = (seed >> 56) as u8;
            }

            let encoded = LossyConfig::new(2.0)
                .with_effort(7)
                .encode(&data, w, h, PixelLayout::Rgb8)
                .unwrap();

            // djxl is the ground-truth decoder
            let djxl = djxl_path_string();
            if std::path::Path::new(&djxl).exists() {
                let tmp = std::env::temp_dir().join(format!("multi_dc_{w}x{h}.jxl"));
                std::fs::write(&tmp, &encoded).unwrap();
                let out = std::env::temp_dir().join(format!("multi_dc_{w}x{h}_dec.png"));
                let result = Command::new(&djxl)
                    .args([&tmp, &out])
                    .output()
                    .expect("djxl failed to run");
                assert!(
                    result.status.success(),
                    "djxl failed for {w}x{h}: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
            }
        }
    }

    /// Test VarDCT lossy + alpha with multi-DC-groups.
    /// Alpha is encoded as a modular extra channel in HfGroup sections.
    #[test]
    fn test_lossy_multi_dc_group_alpha() {
        let (w, h) = (2100, 256);
        let mut data = vec![0u8; w * h * 4]; // RGBA
        let mut seed = 99u64;
        for val in data.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *val = (seed >> 56) as u8;
        }

        let encoded = LossyConfig::new(2.0)
            .with_effort(7)
            .encode(&data, w as u32, h as u32, PixelLayout::Rgba8)
            .unwrap();

        // djxl handles alpha correctly
        let djxl = djxl_path_string();
        if std::path::Path::new(&djxl).exists() {
            let tmp = std::env::temp_dir().join("multi_dc_alpha.jxl");
            std::fs::write(&tmp, &encoded).unwrap();
            let out = std::env::temp_dir().join("multi_dc_alpha_dec.png");
            let result = Command::new(&djxl)
                .args([&tmp, &out])
                .output()
                .expect("djxl failed to run");
            assert!(
                result.status.success(),
                "djxl failed for RGBA {w}x{h}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }

    /// Encode a P3 gradient, decode to linear sRGB, and verify the decoded pixels
    /// match P3→sRGB reference values within lossy tolerance.
    ///
    /// This is the real correctness test: without the primaries matrix, the
    /// decoded sRGB values would match the raw P3 input (wrong), not the
    /// P3→sRGB converted reference (correct).
    #[test]
    fn test_lossy_wide_gamut_p3_roundtrip() {
        use crate::headers::color_encoding::{ColorEncoding, Primaries, TransferFunction};

        // P3→sRGB matrix (same values as in vardct/xyb.rs)
        #[allow(clippy::excessive_precision)]
        const P3_TO_SRGB: [[f32; 3]; 3] = [
            [1.2249401763, -0.2249401763, 0.0000000000],
            [-0.0420569547, 1.0420569547, 0.0000000000],
            [-0.0196375546, -0.0786360456, 1.0982736001],
        ];

        let width = 32u32;
        let height = 32u32;
        let n = (width * height) as usize;

        // Build a gradient that exercises the gamut: R and G ramp, B inverse
        let mut linear_p3 = vec![0.0f32; n * 3];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let i = (y * width as usize + x) * 3;
                linear_p3[i] = x as f32 / 31.0;
                linear_p3[i + 1] = y as f32 / 31.0;
                linear_p3[i + 2] = 0.3 * (1.0 - x as f32 / 31.0);
            }
        }

        // Compute reference: what the decoded sRGB linear values SHOULD be
        let mut reference_srgb = vec![0.0f32; n * 3];
        for i in 0..n {
            let r = linear_p3[i * 3];
            let g = linear_p3[i * 3 + 1];
            let b = linear_p3[i * 3 + 2];
            reference_srgb[i * 3] =
                P3_TO_SRGB[0][0] * r + P3_TO_SRGB[0][1] * g + P3_TO_SRGB[0][2] * b;
            reference_srgb[i * 3 + 1] =
                P3_TO_SRGB[1][0] * r + P3_TO_SRGB[1][1] * g + P3_TO_SRGB[1][2] * b;
            reference_srgb[i * 3 + 2] =
                P3_TO_SRGB[2][0] * r + P3_TO_SRGB[2][1] * g + P3_TO_SRGB[2][2] * b;
        }

        // Encode with P3 primaries
        let p3_encoding = ColorEncoding {
            primaries: Primaries::P3,
            transfer_function: TransferFunction::Srgb,
            ..ColorEncoding::srgb()
        };
        let bytes: &[u8] = bytemuck::cast_slice(&linear_p3);
        let cfg = LossyConfig::new(0.5); // low distance for tight tolerance
        let encoded = cfg
            .encode_request(width, height, PixelLayout::RgbLinearF32)
            .with_color_encoding(p3_encoding)
            .encode(bytes)
            .unwrap();

        // Decode to sRGB linear via jxl-oxide
        let mut image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .expect("jxl-oxide parse failed");
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).expect("jxl-oxide render failed");
        let fb = render.image_all_channels();
        let decoded_f32 = fb.buf(); // flat [R,G,B, R,G,B, ...] f32 array
        let channels = fb.channels();

        // Compare decoded sRGB linear against reference (P3→sRGB converted input)
        let mut max_err_vs_ref = 0.0f32;
        let mut sum_err_vs_ref = 0.0f64;
        let mut max_err_vs_raw = 0.0f32;
        let mut sum_err_vs_raw = 0.0f64;
        let pixels = (width * height) as usize;
        for i in 0..pixels {
            for c in 0..3.min(channels) {
                let decoded = decoded_f32[i * channels + c];
                let ref_val = reference_srgb[i * 3 + c];
                let raw_val = linear_p3[i * 3 + c]; // what you'd get WITHOUT the matrix

                let err_ref = (decoded - ref_val).abs();
                let err_raw = (decoded - raw_val).abs();
                max_err_vs_ref = max_err_vs_ref.max(err_ref);
                sum_err_vs_ref += err_ref as f64;
                max_err_vs_raw = max_err_vs_raw.max(err_raw);
                sum_err_vs_raw += err_raw as f64;
            }
        }
        let count = (pixels * 3) as f64;
        let avg_ref = sum_err_vs_ref / count;
        let avg_raw = sum_err_vs_raw / count;

        eprintln!(
            "P3 roundtrip: decoded vs P3→sRGB reference: max={max_err_vs_ref:.4}, avg={avg_ref:.6}"
        );
        eprintln!(
            "P3 roundtrip: decoded vs raw P3 (wrong):    max={max_err_vs_raw:.4}, avg={avg_raw:.6}"
        );

        // Decoded should be CLOSE to the reference (P3→sRGB), not the raw P3 input
        assert!(
            avg_ref < avg_raw,
            "decoded pixels should be closer to P3→sRGB reference ({avg_ref:.6}) \
             than to raw P3 input ({avg_raw:.6})"
        );
        // At d=0.5, max error vs reference should be well under 0.1
        assert!(
            max_err_vs_ref < 0.1,
            "max error vs P3→sRGB reference too high: {max_err_vs_ref:.4}"
        );
    }

    /// Full-gamut gradient: encode the same linear pixels as P3 vs sRGB,
    /// decode both to linear sRGB, and verify the primaries matrix produces
    /// measurably different (correct) output.
    #[test]
    fn test_lossy_wide_gamut_p3_gradient_difference() {
        use crate::headers::color_encoding::{ColorEncoding, Primaries, TransferFunction};

        let w = 64u32;
        let h = 64u32;
        let n = (w * h) as usize;
        let mut linear = vec![0.0f32; n * 3];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = (y * w as usize + x) * 3;
                linear[i] = x as f32 / 63.0; // R: 0→1
                linear[i + 1] = y as f32 / 63.0; // G: 0→1
                linear[i + 2] = 0.5 * (1.0 - linear[i]); // B: 0.5→0
            }
        }
        let bytes: &[u8] = bytemuck::cast_slice(&linear);

        // Encode with P3 primaries (correct — applies P3→sRGB matrix)
        let p3_enc = ColorEncoding {
            primaries: Primaries::P3,
            transfer_function: TransferFunction::Srgb,
            ..ColorEncoding::srgb()
        };
        let cfg = LossyConfig::new(1.0);
        let encoded_p3 = cfg
            .encode_request(w, h, PixelLayout::RgbLinearF32)
            .with_color_encoding(p3_enc)
            .encode(bytes)
            .unwrap();

        // Encode with default sRGB primaries (wrong — no matrix)
        let encoded_srgb = cfg
            .encode_request(w, h, PixelLayout::RgbLinearF32)
            .encode(bytes)
            .unwrap();

        // Files should differ in size (different XYB values → different quantization)
        assert_ne!(
            encoded_p3.len(),
            encoded_srgb.len(),
            "P3 and sRGB encodings should differ in size"
        );

        // Both should decode successfully
        let djxl = djxl_path_string();
        if std::path::Path::new(&djxl).exists() {
            for (data, name) in [(&encoded_p3, "p3"), (&encoded_srgb, "srgb")] {
                let tmp = std::env::temp_dir().join(format!("gamut_gradient_{name}.jxl"));
                std::fs::write(&tmp, data).unwrap();
                let out = std::env::temp_dir().join(format!("gamut_gradient_{name}_dec.png"));
                let result = Command::new(&djxl)
                    .args([&tmp, &out])
                    .output()
                    .expect("djxl failed to run");
                assert!(
                    result.status.success(),
                    "djxl failed on {name} gradient: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
            }
        }
    }

    /// Verify that BT.2020 primaries roundtrip correctly.
    #[test]
    fn test_lossy_wide_gamut_bt2020_roundtrip() {
        use crate::headers::color_encoding::{ColorEncoding, Primaries, TransferFunction};

        let width = 16u32;
        let height = 16u32;
        let n = (width * height) as usize;
        // Mid-gray in BT.2020 linear — should transform cleanly
        let linear_bt2020 = vec![0.5f32; n * 3];

        let bt2020_encoding = ColorEncoding {
            primaries: Primaries::Bt2100,
            transfer_function: TransferFunction::Srgb,
            ..ColorEncoding::srgb()
        };

        let bytes: &[u8] = bytemuck::cast_slice(&linear_bt2020);
        let cfg = LossyConfig::new(1.0);
        let encoded = cfg
            .encode_request(width, height, PixelLayout::RgbLinearF32)
            .with_color_encoding(bt2020_encoding)
            .encode(bytes)
            .unwrap();

        // Verify djxl decodes
        let djxl = djxl_path_string();
        if std::path::Path::new(&djxl).exists() {
            let tmp = std::env::temp_dir().join("bt2020_test.jxl");
            std::fs::write(&tmp, &encoded).unwrap();
            let out = std::env::temp_dir().join("bt2020_test_dec.png");
            let result = Command::new(&djxl)
                .args([&tmp, &out])
                .output()
                .expect("djxl failed to run");
            assert!(
                result.status.success(),
                "djxl failed on BT.2020 encoded file: {}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }

    // ========== DUAL-DECODER VALIDATION TESTS ==========
    // These tests validate encoded files against BOTH jxl-oxide and djxl

    /// Dual-decoder validation for lossless grayscale encoding
    #[test]
    fn test_dual_decode_lossless_gray() {
        let data = vec![0u8, 64, 128, 192, 255, 100, 50, 200];
        let encoded = LosslessConfig::new()
            .encode(&data, 4, 2, PixelLayout::Gray8)
            .unwrap();
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
        let encoded = LosslessConfig::new()
            .encode(&data, 8, 8, PixelLayout::Rgb8)
            .unwrap();
        validate_dual_decoder(&encoded, 8, 8, "lossless_rgb_8x8");
    }

    /// Dual-decoder validation for lossy VarDCT encoding
    /// Note: VarDCT encoding is WIP and may not pass djxl yet
    #[test]
    fn test_dual_decode_lossy() {
        let mut data = vec![0u8; 16 * 16 * 3];
        for y in 0..16 {
            for x in 0..16 {
                let idx = (y * 16 + x) * 3;
                data[idx] = ((x + y) * 8) as u8;
                data[idx + 1] = ((x * 2) % 256) as u8;
                data[idx + 2] = ((y * 2) % 256) as u8;
            }
        }
        let encoded = LossyConfig::new(1.0)
            .encode(&data, 16, 16, PixelLayout::Rgb8)
            .unwrap();
        // Save for debugging
        let lossy_16x16_path = std::env::temp_dir().join("test_16x16_lossy.jxl");
        std::fs::write(&lossy_16x16_path, &encoded).unwrap();
        eprintln!(
            "Saved {} bytes to {}",
            encoded.len(),
            lossy_16x16_path.display()
        );
        // VarDCT is validated against jxl-oxide only until encoder is complete
        let oxide_result = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&encoded));
        assert!(
            oxide_result.is_ok(),
            "jxl-oxide should decode lossy VarDCT: {:?}",
            oxide_result.err()
        );
        let image = oxide_result.unwrap();
        assert_eq!(image.width(), 16);
        assert_eq!(image.height(), 16);

        // Try to actually render the frame (not just parse headers)
        let _render = image
            .render_frame(0)
            .expect("test_dual_decode_lossy: render failed");

        eprintln!("lossy_16x16: PASSED jxl-oxide (rendered successfully)");
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
        let encoded = LosslessConfig::new()
            .encode(&data, 32, 32, PixelLayout::Rgb8)
            .unwrap();
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
        let encoded = LosslessConfig::new()
            .encode(&data, 16, 16, PixelLayout::Rgb8)
            .unwrap();
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
            let encoded = LossyConfig::new(distance)
                .encode(&data, 8, 8, PixelLayout::Rgb8)
                .unwrap();
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

            // Try to actually render the frame (not just parse headers)
            let _render = image.render_frame(0).unwrap_or_else(|e| {
                panic!(
                    "test_dual_decode_lossy_distances: render failed at distance {}: {}",
                    distance, e
                )
            });
        }
        eprintln!("lossy_distances: PASSED jxl-oxide (rendered successfully)");
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
            let encoded = LosslessConfig::new()
                .encode(&data, w as u32, h as u32, PixelLayout::Rgb8)
                .unwrap();
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
        let encoded = LosslessConfig::new()
            .encode(&data, 256, 256, PixelLayout::Rgb8)
            .unwrap();
        validate_dual_decoder(&encoded, 256, 256, "multi_group_256x256_lossless");

        // Also test lossy with jxl-oxide only
        let lossy_encoded = LossyConfig::new(2.0)
            .encode(&data, 256, 256, PixelLayout::Rgb8)
            .unwrap();
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

    /// Test tree learning with Select predictor at various sizes and efforts.
    /// Validates that the Select predictor (predictor 4) produces correct
    /// predictions matching the JXL spec: p = W + N - NW;
    /// if |p-W| < |p-N| then W else N.
    #[test]
    fn test_tree_learning_select_predictor() {
        // Use gradient data that exercises the Select predictor
        // (tree learning at effort 7+ selects it for some contexts)
        for size in [64, 192, 256] {
            let mut data = vec![0u8; size * size * 3];
            for y in 0..size {
                for x in 0..size {
                    let idx = (y * size + x) * 3;
                    data[idx] = (x & 0xFF) as u8;
                    data[idx + 1] = (y & 0xFF) as u8;
                    data[idx + 2] = ((x + y) % 256) as u8;
                }
            }

            for effort in [6, 7, 8] {
                let encoded = LosslessConfig::new()
                    .with_effort(effort)
                    .with_tree_learning(true)
                    .encode(&data, size as u32, size as u32, PixelLayout::Rgb8)
                    .unwrap();

                let decoded =
                    crate::test_helpers::decode_with_jxl_rs(&encoded).unwrap_or_else(|e| {
                        panic!(
                            "jxl-rs decode failed for {}x{} e{}: {:?}",
                            size, size, effort, e
                        )
                    });

                // Verify pixel-exact roundtrip (decoded pixels are f32 0.0-1.0)
                let decoded_u8: Vec<u8> = decoded
                    .pixels
                    .iter()
                    .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
                    .collect();
                assert_eq!(
                    decoded_u8.len(),
                    data.len(),
                    "decoded length mismatch for {}x{} e{}",
                    size,
                    size,
                    effort
                );
                assert_eq!(
                    &decoded_u8[..],
                    &data[..],
                    "pixel data mismatch for {}x{} e{}",
                    size,
                    size,
                    effort
                );
            }
        }
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

        let encoded = LosslessConfig::new()
            .encode(&data, 300, 300, PixelLayout::Rgb8)
            .unwrap();

        // Check JXL signature
        assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);

        eprintln!("Multi-group 300x300: {} bytes", encoded.len());
        std::fs::write(
            std::env::temp_dir().join("test_multigroup_300.jxl"),
            &encoded,
        )
        .unwrap();

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

        let encoded = LosslessConfig::new()
            .encode(&data, 512, 512, PixelLayout::Rgb8)
            .unwrap();

        // Check JXL signature
        assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);

        eprintln!("Multi-group 512x512: {} bytes", encoded.len());
        std::fs::write(
            std::env::temp_dir().join("test_multigroup_512.jxl"),
            &encoded,
        )
        .unwrap();

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
        crate::skip_without_corpus!();
        // (corpus path resolved via crate::test_helpers::corpus_dir)

        // Test a few representative images from the corpus
        let test_images = [
            ("pngsuite/basn2c08.png", false), // RGB lossless
            ("pngsuite/basn0g08.png", true),  // Gray lossless
        ];

        for (image_path, is_gray) in test_images {
            let path = format!(
                "{}/{}",
                crate::test_helpers::corpus_dir().display(),
                image_path
            );
            if !std::path::Path::new(&path).exists() {
                eprintln!("Skipping {}: not found", image_path);
                continue;
            }

            let img = image::open(&path).unwrap();
            let (w, h) = (img.width() as usize, img.height() as usize);

            let encoded = if is_gray {
                let gray = img.to_luma8();
                LosslessConfig::new()
                    .encode(gray.as_raw(), w as u32, h as u32, PixelLayout::Gray8)
                    .unwrap()
            } else {
                let rgb = img.to_rgb8();
                LosslessConfig::new()
                    .encode(rgb.as_raw(), w as u32, h as u32, PixelLayout::Rgb8)
                    .unwrap()
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

        let encoded = LosslessConfig::new()
            .encode(&data, 2, 2, PixelLayout::Rgb8)
            .unwrap();
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

    /// Test single-group: 256x1 for comparison
    #[test]
    fn test_roundtrip_lossless_rgb_singlegroup_256x1() {
        // Single group 256x1 - for comparison with multi-group
        let data = vec![128u8; 256 * 3];
        validate_lossless_roundtrip_rgb(&data, 256, 1, "rgb_singlegroup_256x1");
    }

    /// Test minimal multi-group: 257x1 (just 2 groups in X direction)
    /// Squeeze disabled: tests multi-group boundary handling, not squeeze.
    /// (jxl-rs has a known boundary bug with 1-pixel squeeze sub-buffers; djxl decodes correctly)
    #[test]
    fn test_roundtrip_lossless_rgb_multigroup_257x1() {
        // Tiny 257x1 image - should be 2 groups (256 + 1 pixels)
        let data = vec![128u8; 257 * 3];
        validate_lossless_roundtrip_rgb_config(
            &data,
            257,
            1,
            "rgb_multigroup_257x1",
            LosslessConfig::new().with_squeeze(false),
        );
    }

    /// Test minimal multi-group: 257x257 solid color (simplest case - all zeros residuals)
    /// Squeeze disabled: tests multi-group boundary handling, not squeeze.
    /// (jxl-rs has a known boundary bug with 1-pixel squeeze sub-buffers; djxl decodes correctly)
    #[test]
    fn test_roundtrip_lossless_rgb_multigroup_257_solid() {
        // Solid gray - all predictions should be exact, residuals all 0
        let data = vec![128u8; 257 * 257 * 3];
        validate_lossless_roundtrip_rgb_config(
            &data,
            257,
            257,
            "rgb_multigroup_257_solid",
            LosslessConfig::new().with_squeeze(false),
        );
    }

    /// Test lossless roundtrip for multi-group RGB (300x300)
    #[test]
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
        crate::skip_without_corpus!();
        // (corpus path resolved via crate::test_helpers::corpus_dir)
        let path = format!(
            "{}/pngsuite/basn2c08.png",
            crate::test_helpers::corpus_dir().display()
        );
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
        crate::skip_without_corpus!();
        // (corpus path resolved via crate::test_helpers::corpus_dir)
        let path = format!(
            "{}/pngsuite/basn0g08.png",
            crate::test_helpers::corpus_dir().display()
        );
        if !std::path::Path::new(&path).exists() {
            eprintln!("Skipping: {} not found", path);
            return;
        }

        let img = image::open(&path).unwrap();
        let gray = img.to_luma8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        validate_lossless_roundtrip_gray(gray.as_raw(), w, h, "corpus_basn0g08");
    }
    #[cfg(test)]
    mod lossy_tests {
        use crate::test_helpers::*;

        #[test]
        fn test_lossy_8x8_checkerboard() {
            // 8x8 checkerboard - simplest VarDCT test
            let mut data = vec![0u8; 8 * 8 * 3];
            for y in 0..8 {
                for x in 0..8 {
                    let idx = (y * 8 + x) * 3;
                    let val = if (x + y) % 2 == 0 { 255 } else { 0 };
                    data[idx] = val; // R
                    data[idx + 1] = val; // G
                    data[idx + 2] = val; // B
                }
            }

            // This will verify encoding mode and fail with clear error if wrong
            test_lossy_roundtrip(&data, 8, 8, 1.0, "lossy_8x8_checkerboard")
                .expect("VarDCT 8x8 should work");
        }
    }
}

/// Quality comparison tests between our encoder and libjxl.
/// These tests compare SSIMULACRA2 scores at the same distance values.
#[cfg(test)]
mod quality_comparison_tests {
    use crate::{LossyConfig, PixelLayout};
    use fast_ssim2::{Rgb, compute_frame_ssimulacra2};
    use std::process::Command;
    use yuvxyb::{ColorPrimaries, TransferCharacteristic};

    // (corpus path resolved via crate::test_helpers::corpus_dir)
    fn libjxl_cjxl() -> String {
        crate::test_helpers::cjxl_path()
    }
    fn libjxl_djxl() -> String {
        crate::test_helpers::djxl_path()
    }

    /// Load a PNG image and return RGB8 data
    fn load_png(path: &str) -> Option<(Vec<u8>, usize, usize)> {
        let img = image::open(path).ok()?;
        let rgb = img.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        Some((rgb.to_vec(), w, h))
    }

    /// Compute SSIMULACRA2 score between original and decoded images
    fn compute_ssim2_score(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
        let orig_f32: Vec<[f32; 3]> = original
            .chunks(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect();
        let dec_f32: Vec<[f32; 3]> = decoded
            .chunks(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect();

        let source = Rgb::new(
            orig_f32,
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let distorted = Rgb::new(
            dec_f32,
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        compute_frame_ssimulacra2(source, distorted).unwrap_or(f64::NAN)
    }

    /// Encode with libjxl and decode back to PNG
    fn encode_decode_libjxl(input: &str, distance: f32) -> Option<Vec<u8>> {
        let jxl_path = std::env::temp_dir().join(format!("libjxl_test_{}.jxl", std::process::id()));
        let out_path = std::env::temp_dir().join(format!("libjxl_test_{}.png", std::process::id()));

        // Encode with cjxl
        let status = Command::new(libjxl_cjxl())
            .arg(input)
            .arg(&jxl_path)
            .arg("-d")
            .arg(format!("{}", distance))
            .output()
            .ok()?;

        if !status.status.success() {
            eprintln!("cjxl failed: {}", String::from_utf8_lossy(&status.stderr));
            return None;
        }

        // Decode with djxl
        let status = Command::new(libjxl_djxl())
            .arg(&jxl_path)
            .arg(&out_path)
            .output()
            .ok()?;

        if !status.status.success() {
            eprintln!("djxl failed: {}", String::from_utf8_lossy(&status.stderr));
            return None;
        }

        // Load decoded PNG
        let img = image::open(&out_path).ok()?;
        let rgb = img.to_rgb8();

        // Cleanup temp files
        let _ = std::fs::remove_file(&jxl_path);
        let _ = std::fs::remove_file(&out_path);

        Some(rgb.to_vec())
    }

    /// Encode with our encoder and decode with jxl-oxide
    fn encode_decode_ours(
        data: &[u8],
        width: usize,
        height: usize,
        distance: f32,
    ) -> Option<Vec<u8>> {
        use crate::{LossyConfig, PixelLayout};

        let encoded = LossyConfig::new(distance)
            .encode(data, width as u32, height as u32, PixelLayout::Rgb8)
            .ok()?;

        // Decode with jxl-oxide
        let img = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .ok()?;

        let frame = img.render_frame(0).ok()?;
        let fb = frame.image_all_channels();
        let buf = fb.buf();
        let channels = fb.channels();

        // Convert f32 back to u8
        let mut decoded = Vec::with_capacity(width * height * 3);
        for i in 0..(width * height) {
            let idx = i * channels;
            decoded.push((buf[idx].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            decoded.push((buf[idx + 1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            decoded.push((buf[idx + 2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }

        Some(decoded)
    }

    /// Get file sizes for comparison
    fn get_encoded_sizes(
        input_path: &str,
        data: &[u8],
        width: usize,
        height: usize,
        distance: f32,
    ) -> (Option<usize>, Option<usize>) {
        use crate::{LossyConfig, PixelLayout};

        // Our encoder size
        let our_size = LossyConfig::new(distance)
            .encode(data, width as u32, height as u32, PixelLayout::Rgb8)
            .ok()
            .map(|e| e.len());

        // libjxl size
        let jxl_path = std::env::temp_dir().join(format!("libjxl_size_{}.jxl", std::process::id()));
        let libjxl_size = Command::new(libjxl_cjxl())
            .arg(input_path)
            .arg(&jxl_path)
            .arg("-d")
            .arg(format!("{}", distance))
            .output()
            .ok()
            .and_then(|status| {
                if status.status.success() {
                    std::fs::metadata(&jxl_path).ok().map(|m| m.len() as usize)
                } else {
                    None
                }
            });

        let _ = std::fs::remove_file(&jxl_path);

        (our_size, libjxl_size)
    }

    #[test]
    fn test_quality_comparison_kodak01() {
        crate::skip_without_corpus!();
        let path = format!(
            "{}/kodak/1.png",
            crate::test_helpers::corpus_dir().display()
        );
        if !std::path::Path::new(&path).exists() {
            eprintln!("Skipping: Kodak image not found at {}", path);
            return;
        }

        let (data, width, height) = load_png(&path).expect("Failed to load image");
        let distances = [0.5, 1.0, 2.0, 4.0];

        eprintln!("\n╔════════════════════════════════════════════════════════════════════════╗");
        eprintln!("║           QUALITY COMPARISON: jxl-encoder-rs vs libjxl               ║");
        eprintln!("║                    Kodak 1 (768x512)                                 ║");
        eprintln!("╠════════════════════════════════════════════════════════════════════════╣");
        eprintln!(
            "║ {:>8} │ {:>12} {:>12} │ {:>10} {:>10} │ {:>8} ║",
            "Distance", "Our SSIM2", "libjxl SSIM2", "Our Size", "libjxl", "Δ SSIM2"
        );
        eprintln!("╠════════════════════════════════════════════════════════════════════════╣");

        for distance in distances {
            // Encode and decode with both
            let our_decoded = encode_decode_ours(&data, width, height, distance);
            let libjxl_decoded = encode_decode_libjxl(&path, distance);

            // Compute SSIMULACRA2 scores
            let our_score = our_decoded
                .as_ref()
                .map(|d| compute_ssim2_score(&data, d, width, height))
                .unwrap_or(f64::NAN);

            let libjxl_score = libjxl_decoded
                .as_ref()
                .map(|d| compute_ssim2_score(&data, d, width, height))
                .unwrap_or(f64::NAN);

            // Get file sizes
            let (our_size, libjxl_size) = get_encoded_sizes(&path, &data, width, height, distance);

            let delta = our_score - libjxl_score;
            let delta_str = if delta.is_nan() {
                "N/A".to_string()
            } else {
                format!("{:+.2}", delta)
            };

            eprintln!(
                "║ {:>8.1} │ {:>12.2} {:>12.2} │ {:>10} {:>10} │ {:>8} ║",
                distance,
                our_score,
                libjxl_score,
                our_size
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "ERR".to_string()),
                libjxl_size
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "ERR".to_string()),
                delta_str
            );
        }

        eprintln!("╚════════════════════════════════════════════════════════════════════════╝");
        eprintln!("\nNote: SSIMULACRA2 scores above 70 are generally good quality.");
        eprintln!("      Δ SSIM2 shows difference (positive = ours is better).");
    }

    /// Comprehensive comparison across multiple Kodak images.
    /// Run with: cargo test quality_comparison_comprehensive -- --nocapture --ignored
    #[test]
    #[ignore]
    fn test_quality_comparison_comprehensive() {
        let kodak_images: Vec<String> = (1..=24)
            .map(|i| {
                format!(
                    "{}/kodak/{}.png",
                    crate::test_helpers::corpus_dir().display(),
                    i
                )
            })
            .filter(|p| std::path::Path::new(p).exists())
            .collect();

        if kodak_images.is_empty() {
            eprintln!("No Kodak images found, skipping comprehensive test");
            return;
        }

        eprintln!(
            "\n╔═══════════════════════════════════════════════════════════════════════════════╗"
        );
        eprintln!(
            "║              COMPREHENSIVE QUALITY COMPARISON (Kodak Suite)                  ║"
        );
        eprintln!(
            "╠═══════════════════════════════════════════════════════════════════════════════╣"
        );

        let distances = [1.0, 2.0, 4.0];

        for distance in distances {
            eprintln!(
                "╠═══════════════════════════════════════════════════════════════════════════════╣"
            );
            eprintln!(
                "║ Distance: {:.1}                                                                 ║",
                distance
            );
            eprintln!(
                "╠═══════════════════════════════════════════════════════════════════════════════╣"
            );
            eprintln!(
                "║ {:>8} │ {:>12} {:>12} │ {:>10} {:>10} │ {:>8} ║",
                "Image", "Our SSIM2", "libjxl SSIM2", "Our Size", "libjxl", "Δ SSIM2"
            );
            eprintln!(
                "╠═══════════════════════════════════════════════════════════════════════════════╣"
            );

            let mut our_scores = Vec::new();
            let mut libjxl_scores = Vec::new();

            for (idx, path) in kodak_images.iter().enumerate() {
                let (data, width, height) = match load_png(path) {
                    Some(d) => d,
                    None => continue,
                };

                let our_decoded = encode_decode_ours(&data, width, height, distance);
                let libjxl_decoded = encode_decode_libjxl(path, distance);

                let our_score = our_decoded
                    .as_ref()
                    .map(|d| compute_ssim2_score(&data, d, width, height))
                    .unwrap_or(f64::NAN);

                let libjxl_score = libjxl_decoded
                    .as_ref()
                    .map(|d| compute_ssim2_score(&data, d, width, height))
                    .unwrap_or(f64::NAN);

                let (our_size, libjxl_size) =
                    get_encoded_sizes(path, &data, width, height, distance);

                if !our_score.is_nan() {
                    our_scores.push(our_score);
                }
                if !libjxl_score.is_nan() {
                    libjxl_scores.push(libjxl_score);
                }

                let delta = our_score - libjxl_score;
                eprintln!(
                    "║ {:>8} │ {:>12.2} {:>12.2} │ {:>10} {:>10} │ {:>+8.2} ║",
                    format!("kodak{:02}", idx + 1),
                    our_score,
                    libjxl_score,
                    our_size
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "ERR".to_string()),
                    libjxl_size
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "ERR".to_string()),
                    delta
                );
            }

            // Print averages
            let our_avg = if our_scores.is_empty() {
                f64::NAN
            } else {
                our_scores.iter().sum::<f64>() / our_scores.len() as f64
            };
            let libjxl_avg = if libjxl_scores.is_empty() {
                f64::NAN
            } else {
                libjxl_scores.iter().sum::<f64>() / libjxl_scores.len() as f64
            };

            eprintln!(
                "╠═══════════════════════════════════════════════════════════════════════════════╣"
            );
            eprintln!(
                "║ {:>8} │ {:>12.2} {:>12.2} │ {:>10} {:>10} │ {:>+8.2} ║",
                "AVERAGE",
                our_avg,
                libjxl_avg,
                "",
                "",
                our_avg - libjxl_avg
            );
        }

        eprintln!(
            "╚═══════════════════════════════════════════════════════════════════════════════╝"
        );
    }

    /// Test that horizontal gradient is properly preserved (not transposed to vertical)
    #[test]
    fn test_lossy_horizontal_gradient_orientation() {
        // Create 8x8 horizontal gradient (varies by column, constant by row)
        let mut data = vec![0u8; 8 * 8 * 3];
        for y in 0..8 {
            for x in 0..8 {
                let idx = (y * 8 + x) * 3;
                let val = (x * 32) as u8; // 0, 32, 64, 96, 128, 160, 192, 224
                data[idx] = val; // R
                data[idx + 1] = val; // G
                data[idx + 2] = val; // B
            }
        }

        // Encode with lossy VarDCT
        let encoded = LossyConfig::new(1.0)
            .encode(&data, 8, 8, PixelLayout::Rgb8)
            .unwrap();
        std::fs::write(
            std::env::temp_dir().join("test_hgrad_orientation.jxl"),
            &encoded,
        )
        .unwrap();

        // Decode with jxl-oxide
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .expect("Failed to parse JXL");
        let frame = image.render_frame(0).expect("Failed to render");
        let fb = frame.image_all_channels();
        let samples: Vec<f32> = fb.buf().to_vec();

        // Check row 0: should be gradient (different values per column)
        let row0: Vec<i32> = (0..8)
            .map(|col| {
                let idx = col * 3; // row 0
                (samples[idx] * 255.0).round() as i32
            })
            .collect();

        // Check col 0: should be constant (same value per row)
        let col0: Vec<i32> = (0..8)
            .map(|row| {
                let idx = row * 8 * 3; // col 0
                (samples[idx] * 255.0).round() as i32
            })
            .collect();

        eprintln!("Input: horizontal gradient [0, 32, 64, 96, 128, 160, 192, 224]");
        eprintln!("Row 0 (should vary): {:?}", row0);
        eprintln!("Col 0 (should be constant): {:?}", col0);

        // Row 0 should have variation (horizontal gradient)
        let row0_variance: f64 = {
            let mean = row0.iter().sum::<i32>() as f64 / 8.0;
            row0.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / 8.0
        };

        // Col 0 should be constant (no variation in vertical direction)
        let col0_variance: f64 = {
            let mean = col0.iter().sum::<i32>() as f64 / 8.0;
            col0.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / 8.0
        };

        eprintln!("Row 0 variance: {:.1}", row0_variance);
        eprintln!("Col 0 variance: {:.1}", col0_variance);

        // For a horizontal gradient:
        // - Row variance should be HIGH (values differ across columns)
        // - Column variance should be LOW (values same across rows)
        // If transposed, these would be reversed.
        assert!(
            row0_variance > col0_variance,
            "Gradient is transposed! Row variance ({:.1}) should be > col variance ({:.1})",
            row0_variance,
            col0_variance
        );
    }
}

/// Dual-decoder validation with butteraugli quality metrics.
/// Tests that both jxl-rs and jxl-oxide produce identical results
/// and that butteraugli scores are reasonable for the encoder distance.
#[cfg(test)]
mod dual_decoder_butteraugli_tests {
    use crate::{LossyConfig, PixelLayout};
    use butteraugli::{ButteraugliParams, butteraugli};
    use imgref::Img;
    use rgb::RGB8;
    use std::io::Cursor;
    use std::process::Command;

    // (corpus path resolved via crate::test_helpers::corpus_dir)

    /// Load a PNG image and return RGB8 data with dimensions
    fn load_png(path: &str) -> Option<(Vec<u8>, usize, usize)> {
        let img = image::open(path).ok()?;
        let rgb = img.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        Some((rgb.to_vec(), w, h))
    }

    /// Decode JXL with jxl-oxide and return RGB8 pixels
    fn decode_with_oxide(jxl_data: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
        let image = jxl_oxide::JxlImage::builder()
            .read(Cursor::new(jxl_data))
            .map_err(|e| format!("jxl-oxide parse error: {:?}", e))?;

        let frame = image
            .render_frame(0)
            .map_err(|e| format!("jxl-oxide render error: {:?}", e))?;

        let fb = frame.image_all_channels();
        let samples: &[f32] = fb.buf();
        let width = image.width() as usize;
        let height = image.height() as usize;

        // Convert f32 RGB to u8 RGB
        let mut rgb8 = Vec::with_capacity(width * height * 3);
        for i in 0..(width * height) {
            let r = (samples[i * 3].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (samples[i * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (samples[i * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
            rgb8.push(r);
            rgb8.push(g);
            rgb8.push(b);
        }

        Ok((rgb8, width, height))
    }

    /// Decode JXL with jxl-rs CLI and return RGB8 pixels
    fn decode_with_jxlrs(jxl_data: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
        // Write JXL to temp file
        let jxl_path = std::env::temp_dir().join(format!("test_jxlrs_{}.jxl", std::process::id()));
        let png_path = std::env::temp_dir().join(format!("test_jxlrs_{}.png", std::process::id()));

        std::fs::write(&jxl_path, jxl_data)
            .map_err(|e| format!("Failed to write temp JXL: {}", e))?;

        // Decode with jxl-rs CLI
        let output = Command::new(crate::test_helpers::jxl_cli_path())
            .args([&jxl_path, &png_path])
            .output()
            .map_err(|e| format!("Failed to run jxl_cli: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Clean up
            let _ = std::fs::remove_file(&jxl_path);
            return Err(format!("jxl_cli failed: {}", stderr));
        }

        // Load the decoded PNG
        let img =
            image::open(&png_path).map_err(|e| format!("Failed to load decoded PNG: {}", e))?;
        let rgb = img.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let pixels = rgb.to_vec();

        // Clean up temp files
        let _ = std::fs::remove_file(&jxl_path);
        let _ = std::fs::remove_file(&png_path);

        Ok((pixels, w, h))
    }

    /// Compute butteraugli score between original and decoded images
    fn compute_butteraugli(
        original: &[u8],
        decoded: &[u8],
        width: usize,
        height: usize,
    ) -> Result<f64, String> {
        if original.len() != decoded.len() {
            return Err(format!(
                "Size mismatch: original {} vs decoded {}",
                original.len(),
                decoded.len()
            ));
        }

        // Convert to RGB8 pixels
        let orig_pixels: Vec<RGB8> = original
            .chunks(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect();
        let dec_pixels: Vec<RGB8> = decoded
            .chunks(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect();

        let img1 = Img::new(orig_pixels, width, height);
        let img2 = Img::new(dec_pixels, width, height);

        let params = ButteraugliParams::default();
        let result = butteraugli(img1.as_ref(), img2.as_ref(), &params)
            .map_err(|e| format!("Butteraugli error: {:?}", e))?;

        Ok(result.score)
    }

    /// Test result for a single encode/decode operation
    #[derive(Debug)]
    struct TestResult {
        image_name: String,
        distance: f32,
        oxide_butteraugli: f64,
        jxlrs_butteraugli: f64,
        score_diff: f64,
        encoded_size: usize,
    }

    /// Run dual-decoder test for a single image at multiple distances
    fn test_image_at_distances(path: &str, distances: &[f32]) -> Vec<Result<TestResult, String>> {
        let image_name = std::path::Path::new(path)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let (original, width, height) = match load_png(path) {
            Some(data) => data,
            None => return vec![Err(format!("Failed to load {}", path))],
        };

        // Skip images smaller than 8x8 (butteraugli minimum)
        if width < 8 || height < 8 {
            return vec![Err(format!(
                "{}: too small ({}x{})",
                image_name, width, height
            ))];
        }

        distances
            .iter()
            .map(|&distance| {
                // Encode
                let encoded = LossyConfig::new(distance)
                    .encode(&original, width as u32, height as u32, PixelLayout::Rgb8)
                    .map_err(|e| format!("{} d={}: encode error: {:?}", image_name, distance, e))?;

                // Decode with both decoders
                let (oxide_decoded, _, _) = decode_with_oxide(&encoded)
                    .map_err(|e| format!("{} d={}: {}", image_name, distance, e))?;

                let (jxlrs_decoded, _, _) = decode_with_jxlrs(&encoded)
                    .map_err(|e| format!("{} d={}: {}", image_name, distance, e))?;

                // Compute butteraugli scores
                let oxide_score = compute_butteraugli(&original, &oxide_decoded, width, height)
                    .map_err(|e| {
                        format!("{} d={}: oxide butteraugli: {}", image_name, distance, e)
                    })?;

                let jxlrs_score = compute_butteraugli(&original, &jxlrs_decoded, width, height)
                    .map_err(|e| {
                        format!("{} d={}: jxlrs butteraugli: {}", image_name, distance, e)
                    })?;

                let score_diff = (oxide_score - jxlrs_score).abs();

                Ok(TestResult {
                    image_name: image_name.clone(),
                    distance,
                    oxide_butteraugli: oxide_score,
                    jxlrs_butteraugli: jxlrs_score,
                    score_diff,
                    encoded_size: encoded.len(),
                })
            })
            .collect()
    }

    /// Main test: sweep corpus images at multiple distance values
    /// Validates that both decoders produce matching butteraugli scores
    #[test]
    #[ignore = "Requires codec-corpus and jxl-rs CLI; run with: cargo test dual_decoder_butteraugli -- --ignored --nocapture"]
    fn test_dual_decoder_butteraugli_sweep() {
        // Check prerequisites
        let corpus = crate::test_helpers::corpus_dir();
        if !corpus.exists() {
            eprintln!("SKIP: corpus not found at {}", corpus.display());
            return;
        }
        let jxlrs_cli = crate::test_helpers::jxl_cli_path();
        if !std::path::Path::new(&jxlrs_cli).exists() {
            eprintln!("SKIP: jxl-rs CLI not found at {}", jxlrs_cli);
            eprintln!("Build with: cd ~/work/jxl-rs && cargo build --release -p jxl_cli");
            return;
        }

        // Test images (subset of corpus for reasonable test time)
        let test_images = [
            "pngsuite/basn2c08.png", // 32x32 RGB
            "pngsuite/basn6a08.png", // 32x32 RGBA
            "kodak/1.png",           // 768x512
            "kodak/2.png",           // 768x512
            "kodak/3.png",           // 768x512
            "kodak/10.png",          // 512x768
        ];

        // Distance values to test
        let distances = [0.5, 1.0, 2.0, 4.0];

        eprintln!(
            "\n╔═══════════════════════════════════════════════════════════════════════════════════════╗"
        );
        eprintln!(
            "║                    DUAL-DECODER BUTTERAUGLI VALIDATION                                ║"
        );
        eprintln!(
            "╠═══════════════════════════════════════════════════════════════════════════════════════╣"
        );
        eprintln!(
            "║ {:25} │ {:6} │ {:10} │ {:10} │ {:8} │ {:8} ║",
            "Image", "Dist", "Oxide BA", "jxl-rs BA", "Diff", "Size"
        );
        eprintln!(
            "╠═══════════════════════════════════════════════════════════════════════════════════════╣"
        );

        let mut all_results: Vec<TestResult> = Vec::new();
        let mut errors: Vec<String> = Vec::new();
        let mut max_diff: f64 = 0.0;

        for image_rel in &test_images {
            let path = format!(
                "{}/{}",
                crate::test_helpers::corpus_dir().display(),
                image_rel
            );
            if !std::path::Path::new(&path).exists() {
                eprintln!("║ {:25} │ SKIP: file not found", image_rel);
                continue;
            }

            let results = test_image_at_distances(&path, &distances);

            for result in results {
                match result {
                    Ok(r) => {
                        eprintln!(
                            "║ {:25} │ {:6.1} │ {:10.4} │ {:10.4} │ {:8.4} │ {:8} ║",
                            r.image_name,
                            r.distance,
                            r.oxide_butteraugli,
                            r.jxlrs_butteraugli,
                            r.score_diff,
                            r.encoded_size
                        );
                        max_diff = max_diff.max(r.score_diff);
                        all_results.push(r);
                    }
                    Err(e) => {
                        eprintln!("║ ERROR: {} ║", e);
                        errors.push(e);
                    }
                }
            }
        }

        eprintln!(
            "╠═══════════════════════════════════════════════════════════════════════════════════════╣"
        );

        // Summary statistics
        if !all_results.is_empty() {
            let avg_oxide: f64 = all_results.iter().map(|r| r.oxide_butteraugli).sum::<f64>()
                / all_results.len() as f64;
            let avg_jxlrs: f64 = all_results.iter().map(|r| r.jxlrs_butteraugli).sum::<f64>()
                / all_results.len() as f64;
            let avg_diff: f64 =
                all_results.iter().map(|r| r.score_diff).sum::<f64>() / all_results.len() as f64;

            eprintln!(
                "║ {:25} │ {:6} │ {:10.4} │ {:10.4} │ {:8.4} │ {:8} ║",
                "AVERAGE", "", avg_oxide, avg_jxlrs, avg_diff, ""
            );
            eprintln!(
                "║ {:25} │ {:6} │ {:10} │ {:10} │ {:8.4} │ {:8} ║",
                "MAX DIFF", "", "", "", max_diff, ""
            );
        }

        eprintln!(
            "╚═══════════════════════════════════════════════════════════════════════════════════════╝"
        );

        // Assertions
        // 1. Both decoders should produce very similar butteraugli scores
        //    (allowing small differences due to floating-point and color conversion)
        const MAX_ALLOWED_DIFF: f64 = 0.1;
        assert!(
            max_diff < MAX_ALLOWED_DIFF,
            "Decoder outputs differ too much! Max butteraugli diff: {:.4} (allowed: {:.4})",
            max_diff,
            MAX_ALLOWED_DIFF
        );

        // 2. Check that butteraugli scores are reasonable for the distances
        //    Higher distance = higher butteraugli (more distortion)
        for r in &all_results {
            // For distance >= 1.0, butteraugli should be roughly in the same ballpark
            // This is a sanity check, not a strict requirement
            if r.distance >= 1.0 && r.oxide_butteraugli > r.distance as f64 * 5.0 {
                eprintln!(
                    "WARNING: {} d={} has unexpectedly high butteraugli: {:.4}",
                    r.image_name, r.distance, r.oxide_butteraugli
                );
            }
        }

        // 3. No errors should have occurred
        assert!(
            errors.is_empty(),
            "Encountered {} errors during testing:\n{}",
            errors.len(),
            errors.join("\n")
        );

        eprintln!("\n✓ All {} test cases passed", all_results.len());
    }

    /// Quick sanity test with a single synthetic image
    #[test]
    #[ignore = "Requires jxl-rs CLI; run with: cargo test test_dual_decoder_butteraugli_quick -- --ignored --nocapture"]
    fn test_dual_decoder_butteraugli_quick() {
        let jxlrs_cli = crate::test_helpers::jxl_cli_path();
        if !std::path::Path::new(&jxlrs_cli).exists() {
            eprintln!("SKIP: jxl-rs CLI not found at {}", jxlrs_cli);
            return;
        }

        // Create a simple gradient test image
        let width = 64;
        let height = 64;
        let mut original = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                original[idx] = (x * 4) as u8; // R: horizontal gradient
                original[idx + 1] = (y * 4) as u8; // G: vertical gradient
                original[idx + 2] = 128; // B: constant
            }
        }

        let distances = [1.0f32, 2.0];

        for distance in distances {
            let encoded = LossyConfig::new(distance)
                .encode(&original, width as u32, height as u32, PixelLayout::Rgb8)
                .expect("Encode failed");

            let (oxide_decoded, _, _) =
                decode_with_oxide(&encoded).expect("jxl-oxide decode failed");
            let (jxlrs_decoded, _, _) = decode_with_jxlrs(&encoded).expect("jxl-rs decode failed");

            let oxide_score = compute_butteraugli(&original, &oxide_decoded, width, height)
                .expect("Butteraugli failed");
            let jxlrs_score = compute_butteraugli(&original, &jxlrs_decoded, width, height)
                .expect("Butteraugli failed");

            let diff = (oxide_score - jxlrs_score).abs();

            eprintln!(
                "Distance {:.1}: oxide={:.4}, jxl-rs={:.4}, diff={:.4}",
                distance, oxide_score, jxlrs_score, diff
            );

            assert!(
                diff < 0.1,
                "Decoder outputs differ! oxide={:.4}, jxl-rs={:.4}, diff={:.4}",
                oxide_score,
                jxlrs_score,
                diff
            );
        }

        eprintln!("✓ Quick butteraugli test passed");
    }

    /// Comprehensive corpus test for CLIC and CID22 datasets
    /// Tests encode/decode across all images - supports resume via CSV file
    #[test]
    #[ignore = "Full corpus test; run with: cargo test test_corpus_clic_cid -- --ignored --nocapture"]
    fn test_corpus_clic_cid() {
        use std::collections::HashSet;
        use std::fs::{File, OpenOptions};
        use std::io::{BufRead, BufReader, Write};
        use std::time::Instant;

        let corpus_path = crate::test_helpers::corpus_dir();
        if !corpus_path.exists() {
            println!("SKIP: corpus not found at {}", corpus_path.display());
            return;
        }

        // Results file for resume support
        let results_path = std::env::temp_dir().join("jxl_corpus_results.csv");
        let failures_path = std::env::temp_dir().join("jxl_corpus_failures.txt");

        // Load previously processed images
        let mut processed: HashSet<String> = HashSet::new();
        let mut prev_encode_ok = 0usize;
        let mut prev_decode_ok = 0usize;
        let mut prev_total_size = 0usize;
        let mut prev_total_pixels = 0usize;

        if let Ok(file) = File::open(&results_path) {
            let reader = BufReader::new(file);
            for line in reader.lines().skip(1).flatten() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 5 {
                    processed.insert(parts[0].to_string());
                    prev_encode_ok += 1;
                    prev_decode_ok += 1;
                    if let (Ok(w), Ok(h), Ok(size)) = (
                        parts[1].parse::<usize>(),
                        parts[2].parse::<usize>(),
                        parts[4].parse::<usize>(),
                    ) {
                        prev_total_pixels += w * h;
                        prev_total_size += size;
                    }
                }
            }
        }

        // Collect all PNG files from CID22 and clic2025
        let mut images: Vec<std::path::PathBuf> = Vec::new();

        let cid_path = corpus_path.join("CID22");
        if cid_path.exists() {
            for entry in walkdir::WalkDir::new(&cid_path)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if path.extension().map(|e| e == "png").unwrap_or(false) {
                    images.push(path.to_path_buf());
                }
            }
        }

        let clic_path = corpus_path.join("clic2025");
        if clic_path.exists() {
            for entry in walkdir::WalkDir::new(&clic_path)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if path.extension().map(|e| e == "png").unwrap_or(false) {
                    images.push(path.to_path_buf());
                }
            }
        }

        println!("\n=== CORPUS TEST: CLIC + CID22 ===");
        println!(
            "Found {} images, {} already processed",
            images.len(),
            processed.len()
        );

        // Open results file for appending
        let mut results_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&results_path)
            .expect("Failed to open results file");

        // Write header if file is empty
        if processed.is_empty() {
            writeln!(results_file, "path,width,height,status,size").unwrap();
        }

        let mut failures_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&failures_path)
            .expect("Failed to open failures file");

        let mut encode_ok = prev_encode_ok;
        let mut decode_ok = prev_decode_ok;
        let mut total_size = prev_total_size;
        let mut total_pixels = prev_total_pixels;
        let mut new_tests = 0;

        let start = Instant::now();

        for image_path in images.iter() {
            let rel_path = image_path
                .strip_prefix(&corpus_path)
                .unwrap_or(image_path)
                .to_string_lossy()
                .to_string();

            // Skip already processed
            if processed.contains(&rel_path) {
                continue;
            }

            // Load image
            let (original, width, height) = match load_png(image_path.to_str().unwrap()) {
                Some(data) => data,
                None => {
                    writeln!(failures_file, "{}: failed to load", rel_path).unwrap();
                    continue;
                }
            };

            // Skip very small images
            if width < 8 || height < 8 {
                writeln!(results_file, "{},{},{},skipped,0", rel_path, width, height).unwrap();
                continue;
            }

            new_tests += 1;

            // Encode
            let encoded = match LossyConfig::new(1.0).encode(
                &original,
                width as u32,
                height as u32,
                PixelLayout::Rgb8,
            ) {
                Ok(data) => {
                    encode_ok += 1;
                    data
                }
                Err(e) => {
                    writeln!(
                        failures_file,
                        "{} ({}x{}): ENCODE FAIL: {:?}",
                        rel_path, width, height, e
                    )
                    .unwrap();
                    writeln!(
                        results_file,
                        "{},{},{},encode_fail,0",
                        rel_path, width, height
                    )
                    .unwrap();
                    continue;
                }
            };

            // Decode with jxl-oxide
            match decode_with_oxide(&encoded) {
                Ok(_) => {
                    decode_ok += 1;
                    total_size += encoded.len();
                    total_pixels += width * height;
                    writeln!(
                        results_file,
                        "{},{},{},ok,{}",
                        rel_path,
                        width,
                        height,
                        encoded.len()
                    )
                    .unwrap();
                }
                Err(e) => {
                    writeln!(
                        failures_file,
                        "{} ({}x{}): DECODE FAIL: {}",
                        rel_path, width, height, e
                    )
                    .unwrap();
                    writeln!(
                        results_file,
                        "{},{},{},decode_fail,{}",
                        rel_path,
                        width,
                        height,
                        encoded.len()
                    )
                    .unwrap();
                }
            };

            // Progress every 10 new images
            if new_tests % 10 == 0 {
                let total_done = processed.len() + new_tests;
                println!(
                    "Progress: {}/{} ({:.1}%) - encode:{} decode:{} - {:.1}s",
                    total_done,
                    images.len(),
                    total_done as f64 / images.len() as f64 * 100.0,
                    encode_ok,
                    decode_ok,
                    start.elapsed().as_secs_f64()
                );
                // Flush to ensure we don't lose progress
                results_file.flush().unwrap();
                failures_file.flush().unwrap();
            }
        }

        let elapsed = start.elapsed();
        let total_tests = encode_ok.max(1);

        // Summary
        println!("\n=== SUMMARY ===");
        println!("Total processed: {}", total_tests);
        println!(
            "Encode success:  {} ({:.1}%)",
            encode_ok,
            encode_ok as f64 / total_tests as f64 * 100.0
        );
        println!(
            "Decode success:  {} ({:.1}%)",
            decode_ok,
            decode_ok as f64 / total_tests as f64 * 100.0
        );
        println!("Time elapsed:    {:.1}s", elapsed.as_secs_f64());

        if total_pixels > 0 {
            let bpp = total_size as f64 * 8.0 / total_pixels as f64;
            println!(
                "Total size:      {:.2} MB",
                total_size as f64 / 1024.0 / 1024.0
            );
            println!("Avg bpp:         {:.3}", bpp);
        }

        println!("\nResults: {}", results_path.display());
        println!("Failures: {}", failures_path.display());

        // Assert high success rate
        let success_rate = decode_ok as f64 / total_tests as f64 * 100.0;
        assert!(
            success_rate >= 95.0,
            "Decode success rate {:.1}% is below 95% threshold",
            success_rate
        );
    }

    /// Quick quality check on a few corpus images using butteraugli
    /// Run with: cargo test test_corpus_quality_sample -- --ignored --nocapture
    #[test]
    #[ignore = "Quality sampling test"]
    fn test_corpus_quality_sample() {
        let corpus_path = crate::test_helpers::corpus_dir();
        if !corpus_path.exists() {
            eprintln!("SKIP: codec-corpus not found");
            return;
        }

        // Sample diverse images
        let samples = [
            "CID22/CID22-512/training/258947.png",
            "CID22/CID22-512/training/1183021.png",
            "CID22/CID22-512/training/pexels-photo-4210863.png",
        ];

        println!("\n=== CORPUS QUALITY SAMPLE (Butteraugli) ===");
        println!(
            "{:<50} {:>8} {:>12} {:>10}",
            "Image", "Size", "Butteraugli", "Status"
        );
        println!("{}", "-".repeat(85));

        for sample in &samples {
            let path = corpus_path.join(sample);
            if !path.exists() {
                println!("{:<50} SKIP (not found)", sample);
                continue;
            }

            let Some((original, width, height)) = load_png(path.to_str().unwrap()) else {
                println!("{:<50} Load error", sample);
                continue;
            };

            // Encode at distance 1.0
            let Ok(encoded) = LossyConfig::new(1.0).encode(
                &original,
                width as u32,
                height as u32,
                PixelLayout::Rgb8,
            ) else {
                println!("{:<50} Encode error", sample);
                continue;
            };

            // Decode with jxl-oxide
            let Ok((decoded, _, _)) = decode_with_oxide(&encoded) else {
                println!("{:<50} Decode error", sample);
                continue;
            };

            match compute_butteraugli(&original, &decoded, width, height) {
                Ok(score) => {
                    let status = if score < 1.0 {
                        "EXCELLENT"
                    } else if score < 2.0 {
                        "GOOD"
                    } else if score < 4.0 {
                        "FAIR"
                    } else {
                        "POOR"
                    };
                    println!(
                        "{:<50} {:>6}KB {:>12.4} {:>10}",
                        sample,
                        encoded.len() / 1024,
                        score,
                        status
                    );
                }
                Err(e) => println!("{:<50} Butteraugli error: {}", sample, e),
            }
        }
    }

    /// Save a broken image for visual comparison
    #[test]
    #[ignore = "Visual comparison test"]
    fn test_save_broken_image() {
        let original_path = format!(
            "{}/clic2025/validation/097cb426910ba8ce2525dd8bb7fb1777.png",
            crate::test_helpers::corpus_dir().display()
        );

        let Some((original, width, height)) = load_png(&original_path) else {
            panic!("Failed to load {}", original_path);
        };

        println!("Loaded {}x{} image", width, height);

        // Encode
        let encoded = LossyConfig::new(1.0)
            .encode(&original, width as u32, height as u32, PixelLayout::Rgb8)
            .expect("Encode failed");

        // Save JXL
        let broken_jxl_path = std::env::temp_dir().join("broken.jxl");
        std::fs::write(&broken_jxl_path, &encoded).unwrap();
        println!(
            "Saved {} ({} bytes)",
            broken_jxl_path.display(),
            encoded.len()
        );

        // Decode with jxl-oxide
        let jxl_image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .expect("Decode failed");

        let frame = jxl_image.render_frame(0).expect("Render failed");
        let fb = frame.image_all_channels();
        let buf = fb.buf();
        let channels = fb.channels();

        // Convert to RGB8
        let mut decoded = vec![0u8; width * height * 3];
        for i in 0..(width * height) {
            let idx = i * channels;
            decoded[i * 3] = (buf[idx].clamp(0.0, 1.0) * 255.0).round() as u8;
            decoded[i * 3 + 1] = (buf[idx + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            decoded[i * 3 + 2] = (buf[idx + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
        }

        // Save decoded PNG
        let broken_decoded_path = std::env::temp_dir().join("broken_decoded.png");
        image::save_buffer(
            &broken_decoded_path,
            &decoded,
            width as u32,
            height as u32,
            image::ColorType::Rgb8,
        )
        .expect("Failed to save decoded PNG");

        println!("Saved {}", broken_decoded_path.display());
        println!("\nRun:");
        println!("  display {} &", original_path);
        println!("  display {} &", broken_decoded_path.display());
    }

    /// Test quality on frymire.png - a real photo that catches bugs synthetic images miss.
    ///
    /// This test is MANDATORY for quality validation. Synthetic images mask bugs like
    /// raw_quant=1 where synthetic tests show SSIM2 63-85 but real photos get SSIM2 23.
    #[test]
    #[ignore = "Real photo quality test - run with: cargo test test_frymire_quality -- --ignored --nocapture"]
    fn test_frymire_quality() {
        // frymire.png is stored in jxl_encoder/tests/images/
        let frymire_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/images/frymire.png");

        let Some((original, width, height)) = load_png(frymire_path) else {
            panic!("Failed to load frymire.png from {}", frymire_path);
        };

        println!("Loaded frymire.png: {}x{}", width, height);

        // Encode at distance 1.0
        let encoded = LossyConfig::new(1.0)
            .encode(&original, width as u32, height as u32, PixelLayout::Rgb8)
            .expect("Encode failed");
        println!(
            "Encoded: {} bytes ({:.2} bpp)",
            encoded.len(),
            encoded.len() as f64 * 8.0 / (width * height) as f64
        );

        // Decode with jxl-oxide
        let jxl_image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .expect("JXL parse failed");

        let frame = jxl_image.render_frame(0).expect("Render failed");
        let fb = frame.image_all_channels();
        let buf = fb.buf();
        let channels = fb.channels();

        // Convert to RGB8 for SSIM2
        let mut decoded = vec![0u8; width * height * 3];
        for i in 0..(width * height) {
            let idx = i * channels;
            decoded[i * 3] = (buf[idx].clamp(0.0, 1.0) * 255.0).round() as u8;
            decoded[i * 3 + 1] = (buf[idx + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            decoded[i * 3 + 2] = (buf[idx + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
        }

        // Compute SSIM2 using fast-ssim2
        use fast_ssim2::compute_ssimulacra2;
        use imgref::ImgVec;

        // Convert to [u8; 3] arrays
        let original_rgb: Vec<[u8; 3]> = original
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let decoded_rgb: Vec<[u8; 3]> = decoded
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();

        let original_img = ImgVec::new(original_rgb, width, height);
        let decoded_img = ImgVec::new(decoded_rgb, width, height);

        let ssim2 = compute_ssimulacra2(original_img.as_ref(), decoded_img.as_ref())
            .expect("SSIM2 computation failed");
        println!("SSIM2: {:.2}", ssim2);

        // libjxl at d=1.0 achieves SSIM2 ~80+ on real photos
        // Our target is SSIM2 > 70 (accounting for implementation differences)
        const MIN_SSIM2: f64 = 70.0;

        if ssim2 < MIN_SSIM2 {
            // Save files for debugging
            let frymire_jxl = std::env::temp_dir().join("frymire.jxl");
            let frymire_decoded = std::env::temp_dir().join("frymire_decoded.png");
            std::fs::write(&frymire_jxl, &encoded).ok();
            image::save_buffer(
                &frymire_decoded,
                &decoded,
                width as u32,
                height as u32,
                image::ColorType::Rgb8,
            )
            .ok();
            println!(
                "\nSaved {} and {} for debugging",
                frymire_jxl.display(),
                frymire_decoded.display()
            );
        }

        assert!(
            ssim2 >= MIN_SSIM2,
            "SSIM2 {:.2} below minimum {:.2} - real photo quality broken!\n\
             This test catches bugs that synthetic images miss (like raw_quant=1).\n\
             See CLAUDE.md 'Known Bugs' section.",
            ssim2,
            MIN_SSIM2
        );

        println!("PASS: SSIM2 {:.2} >= {:.2}", ssim2, MIN_SSIM2);
    }
}

mod tree_learning_tests {
    use crate::{LosslessConfig, PixelLayout};

    /// Helper: encode RGB with tree learning enabled, decode with jxl-rs, verify lossless.
    fn validate_tree_learning_roundtrip_rgb(
        original: &[u8],
        width: usize,
        height: usize,
        test_name: &str,
    ) {
        assert_eq!(original.len(), width * height * 3);

        let encoded = LosslessConfig::new()
            .with_tree_learning(true)
            .encode(original, width as u32, height as u32, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("{}: encoding failed: {}", test_name, e));

        let path = std::env::temp_dir().join(format!("{}.jxl", test_name));
        let _ = std::fs::write(&path, &encoded);
        eprintln!(
            "{}: Saved {} bytes to {}",
            test_name,
            encoded.len(),
            path.display()
        );

        // Decode with jxl-rs (PRIMARY decoder)
        let decoded_img = crate::test_helpers::decode_with_jxl_rs(&encoded)
            .unwrap_or_else(|e| panic!("{}: jxl-rs decode failed: {}", test_name, e));

        assert_eq!(decoded_img.width, width, "{}: width mismatch", test_name);
        assert_eq!(decoded_img.height, height, "{}: height mismatch", test_name);

        let decoded: Vec<u8> = decoded_img
            .pixels
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        assert_eq!(
            decoded.len(),
            original.len(),
            "{}: decoded size mismatch",
            test_name
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
                        "{}: pixel {} ch {} differs: {} vs {} (diff={})",
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
            "{}: PASSED tree learning roundtrip ({}x{}, {} bytes)",
            test_name,
            width,
            height,
            encoded.len()
        );
    }

    /// Helper: encode grayscale with tree learning enabled, decode with jxl-rs, verify lossless.
    fn validate_tree_learning_roundtrip_gray(
        original: &[u8],
        width: usize,
        height: usize,
        test_name: &str,
    ) {
        assert_eq!(original.len(), width * height);

        let encoded = LosslessConfig::new()
            .with_tree_learning(true)
            .encode(original, width as u32, height as u32, PixelLayout::Gray8)
            .unwrap_or_else(|e| panic!("{}: encoding failed: {}", test_name, e));

        let path = std::env::temp_dir().join(format!("{}.jxl", test_name));
        let _ = std::fs::write(&path, &encoded);
        eprintln!(
            "{}: Saved {} bytes to {}",
            test_name,
            encoded.len(),
            path.display()
        );

        // Decode with jxl-rs (PRIMARY decoder)
        let decoded_img = crate::test_helpers::decode_with_jxl_rs(&encoded)
            .unwrap_or_else(|e| panic!("{}: jxl-rs decode failed: {}", test_name, e));

        assert_eq!(decoded_img.width, width, "{}: width mismatch", test_name);
        assert_eq!(decoded_img.height, height, "{}: height mismatch", test_name);

        // jxl-rs returns all channels; for grayscale just take first component per pixel
        let decoded: Vec<u8> = decoded_img
            .pixels
            .iter()
            .step_by(decoded_img.channels)
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

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
                    eprintln!(
                        "{}: pixel {} differs: {} vs {} (diff={})",
                        test_name, i, orig, dec, diff
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
            "{}: PASSED tree learning roundtrip ({}x{}, {} bytes)",
            test_name,
            width,
            height,
            encoded.len()
        );
    }

    #[test]
    fn test_tree_learning_gray_constant_8x8() {
        // Constant image: tree learning should produce a valid single-leaf tree
        let data = vec![128u8; 8 * 8];
        validate_tree_learning_roundtrip_gray(&data, 8, 8, "tree_gray_const_8x8");
    }

    #[test]
    fn test_tree_learning_gray_gradient_8x8() {
        let data: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        validate_tree_learning_roundtrip_gray(&data, 8, 8, "tree_gray_grad_8x8");
    }

    /// Minimal multi-context test: left/right split
    #[test]
    fn test_tree_learning_gray_leftright_8x8() {
        let mut data = vec![0u8; 8 * 8];
        for y in 0..8 {
            for x in 0..8 {
                data[y * 8 + x] = if x < 4 { 0 } else { 200 };
            }
        }
        validate_tree_learning_roundtrip_gray(&data, 8, 8, "tree_gray_leftright_8x8");
    }

    #[test]
    fn test_tree_learning_gray_32x32() {
        // Larger grayscale with varied content to exercise tree splitting
        let mut data = vec![0u8; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                data[y * 32 + x] = ((x * 8 + y * 3) % 256) as u8;
            }
        }
        validate_tree_learning_roundtrip_gray(&data, 32, 32, "tree_gray_32x32");
    }

    #[test]
    fn test_tree_learning_gray_gradient_128x128() {
        let mut data = vec![0u8; 128 * 128];
        for y in 0..128 {
            for x in 0..128 {
                data[y * 128 + x] = ((x * 2 + y * 3) % 256) as u8;
            }
        }
        validate_tree_learning_roundtrip_gray(&data, 128, 128, "tree_gray_grad_128x128");
    }

    #[test]
    fn test_tree_learning_gray_gradient_48x48() {
        let mut data = vec![0u8; 48 * 48];
        for y in 0..48 {
            for x in 0..48 {
                data[y * 48 + x] = ((x * 8 + y * 3) % 256) as u8;
            }
        }
        validate_tree_learning_roundtrip_gray(&data, 48, 48, "tree_gray_grad_48x48");
    }

    #[test]
    fn test_tree_learning_gray_gradient_64x64() {
        let mut data = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                data[y * 64 + x] = ((x * 4 + y * 3) % 256) as u8;
            }
        }
        validate_tree_learning_roundtrip_gray(&data, 64, 64, "tree_gray_grad_64x64");
    }

    #[test]
    fn test_tree_learning_rgb_checkerboard_8x8() {
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
        validate_tree_learning_roundtrip_rgb(&data, 8, 8, "tree_rgb_checker_8x8");
    }

    #[test]
    fn test_tree_learning_rgb_gradient_32x32_rgb() {
        let size = 32;
        let mut data = vec![0u8; size * size * 3];
        for y in 0..size {
            for x in 0..size {
                let idx = (y * size + x) * 3;
                data[idx] = (x * 8) as u8;
                data[idx + 1] = (y * 8) as u8;
                data[idx + 2] = ((x + y) * 4 % 256) as u8;
            }
        }
        validate_tree_learning_roundtrip_rgb(&data, size, size, "tree_rgb_grad_32x32");
    }

    #[test]
    fn test_tree_learning_rgb_gradient_64x64() {
        let size = 64;
        let mut data = vec![0u8; size * size * 3];
        for y in 0..size {
            for x in 0..size {
                let idx = (y * size + x) * 3;
                data[idx] = (x * 4) as u8;
                data[idx + 1] = (y * 4) as u8;
                data[idx + 2] = ((x + y) * 2 % 256) as u8;
            }
        }
        validate_tree_learning_roundtrip_rgb(&data, size, size, "tree_rgb_grad_64x64");
    }

    #[test]
    fn test_tree_learning_rgb_gradient_128x128() {
        let mut data = vec![0u8; 128 * 128 * 3];
        for y in 0..128 {
            for x in 0..128 {
                let idx = (y * 128 + x) * 3;
                data[idx] = (x * 2) as u8;
                data[idx + 1] = (y * 2) as u8;
                data[idx + 2] = ((x + y) % 256) as u8;
            }
        }
        validate_tree_learning_roundtrip_rgb(&data, 128, 128, "tree_rgb_grad_128x128");
    }

    #[test]
    fn test_tree_learning_rgb_multigroup_300x300() {
        // Multi-group image: 300x300 requires 4 groups (2x2)
        let mut data = vec![0u8; 300 * 300 * 3];
        for y in 0..300 {
            for x in 0..300 {
                let idx = (y * 300 + x) * 3;
                data[idx] = ((x + y) % 256) as u8;
                data[idx + 1] = (x % 256) as u8;
                data[idx + 2] = (y % 256) as u8;
            }
        }
        validate_tree_learning_roundtrip_rgb(&data, 300, 300, "tree_rgb_multi_300x300");
    }

    /// Tree learning + palette auto-detect for few-color RGB image.
    #[test]
    fn test_tree_learning_palette_rgb_4_colors_32x32() {
        let colors: [[u8; 3]; 4] = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]];
        let mut data = vec![0u8; 32 * 32 * 3];
        for y in 0..32 {
            for x in 0..32 {
                let c = &colors[(y / 16 * 2 + x / 16) % 4];
                let i = (y * 32 + x) * 3;
                data[i] = c[0];
                data[i + 1] = c[1];
                data[i + 2] = c[2];
            }
        }
        validate_tree_learning_roundtrip_rgb(&data, 32, 32, "tree_palette_4colors_32x32");
    }
}

// ===== Palette transform roundtrip tests =====

/// Validate palette encoding roundtrip: encode → decode with jxl-rs → pixel-exact match.
fn validate_palette_roundtrip_rgb(data: &[u8], width: usize, height: usize, test_name: &str) {
    let encoded = LosslessConfig::new()
        .encode(data, width as u32, height as u32, PixelLayout::Rgb8)
        .unwrap_or_else(|e| panic!("{}: encoding failed: {}", test_name, e));

    crate::test_helpers::save_test_output("palette", &format!("{test_name}.jxl"), &encoded);

    // Decode with jxl-rs
    let decoded_img = crate::test_helpers::decode_with_jxl_rs(&encoded)
        .unwrap_or_else(|e| panic!("{}: jxl-rs decode failed: {}", test_name, e));

    assert_eq!(decoded_img.width, width, "{}: width mismatch", test_name);
    assert_eq!(decoded_img.height, height, "{}: height mismatch", test_name);

    // Convert f32 to u8
    let decoded: Vec<u8> = decoded_img
        .pixels
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();

    // Pixel-exact match for lossless
    let mut max_diff = 0u8;
    let mut diff_count = 0;
    for (i, (&orig, &dec)) in data.iter().zip(decoded.iter()).enumerate() {
        let diff = (orig as i16 - dec as i16).unsigned_abs() as u8;
        if diff > 0 {
            diff_count += 1;
            if diff > max_diff {
                max_diff = diff;
                let px = i / 3;
                let ch = i % 3;
                eprintln!(
                    "{}: first diff at pixel {} ch {}: orig={} dec={}",
                    test_name, px, ch, orig, dec
                );
            }
        }
    }
    assert_eq!(
        diff_count, 0,
        "{}: {} pixels differ, max_diff={}",
        test_name, diff_count, max_diff
    );
    eprintln!("{}: PASS (pixel-exact)", test_name);
}

#[test]
fn test_palette_roundtrip_2_colors_4x4() {
    // 4x4 image with only 2 colors: red and blue
    let mut data = vec![0u8; 4 * 4 * 3];
    for y in 0..4 {
        for x in 0..4 {
            let idx = (y * 4 + x) * 3;
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
    validate_palette_roundtrip_rgb(&data, 4, 4, "palette_2colors_4x4");
}

#[test]
fn test_palette_roundtrip_8_colors_16x16() {
    // 16x16 image with 8 colors
    let colors: [[u8; 3]; 8] = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [0, 0, 0],
        [255, 255, 255],
    ];
    let mut data = vec![0u8; 16 * 16 * 3];
    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            let c = &colors[(x + y * 3) % 8];
            data[idx] = c[0];
            data[idx + 1] = c[1];
            data[idx + 2] = c[2];
        }
    }
    validate_palette_roundtrip_rgb(&data, 16, 16, "palette_8colors_16x16");
}

#[test]
fn test_palette_roundtrip_64x64() {
    // 64x64 image with 4 colors — larger test
    let colors: [[u8; 3]; 4] = [[10, 20, 30], [100, 150, 200], [200, 50, 75], [30, 180, 90]];
    let mut data = vec![0u8; 64 * 64 * 3];
    for y in 0..64 {
        for x in 0..64 {
            let idx = (y * 64 + x) * 3;
            let c = &colors[(x / 16 + y / 16 * 2) % 4];
            data[idx] = c[0];
            data[idx + 1] = c[1];
            data[idx + 2] = c[2];
        }
    }
    validate_palette_roundtrip_rgb(&data, 64, 64, "palette_4colors_64x64");
}

// ===== Squeeze transform roundtrip tests =====

/// Uses Encoder pipeline with use_squeeze=true, jxl-rs to decode.
#[test]
fn test_squeeze_roundtrip_gray_16x16() {
    use crate::{LosslessConfig, PixelLayout};

    let mut data = vec![0u8; 16 * 16];
    for y in 0..16 {
        for x in 0..16 {
            data[y * 16 + x] = (x * 16 + y * 8) as u8;
        }
    }

    let bytes = LosslessConfig::new()
        .with_squeeze(true)
        .encode(&data, 16, 16, PixelLayout::Gray8)
        .unwrap();

    crate::test_helpers::save_test_output("squeeze", "squeeze_gray_16x16.jxl", &bytes);

    // Decode with jxl-rs
    let decoded_img =
        crate::test_helpers::decode_with_jxl_rs(&bytes).expect("jxl-rs decode failed");
    assert_eq!(decoded_img.width, 16);
    assert_eq!(decoded_img.height, 16);
    let decoded: Vec<u8> = decoded_img
        .pixels
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();
    let mut diff_count = 0;
    let mut max_diff = 0u8;
    for (i, (&orig, &dec)) in data.iter().zip(decoded.iter()).enumerate() {
        let diff = (orig as i16 - dec as i16).unsigned_abs() as u8;
        if diff > max_diff {
            max_diff = diff;
            eprintln!("  pixel {}: orig={} dec={} diff={}", i, orig, dec, diff);
        }
        if diff > 0 {
            diff_count += 1;
        }
    }
    assert_eq!(
        diff_count, 0,
        "Squeeze gray 16x16: {} pixels differ, max_diff={}",
        diff_count, max_diff
    );
    eprintln!("Squeeze gray 16x16: PASS (pixel-exact)");
}

/// Squeeze roundtrip for RGB 32x32 image.
#[test]
fn test_squeeze_roundtrip_rgb_32x32() {
    use crate::{LosslessConfig, PixelLayout};

    let mut data = vec![0u8; 32 * 32 * 3];
    for y in 0..32 {
        for x in 0..32 {
            let i = (y * 32 + x) * 3;
            data[i] = (x * 8) as u8;
            data[i + 1] = (y * 8) as u8;
            data[i + 2] = ((x + y) * 4) as u8;
        }
    }

    let bytes = LosslessConfig::new()
        .with_squeeze(true)
        .encode(&data, 32, 32, PixelLayout::Rgb8)
        .unwrap();

    crate::test_helpers::save_test_output("squeeze", "squeeze_rgb_32x32.jxl", &bytes);

    let decoded_img =
        crate::test_helpers::decode_with_jxl_rs(&bytes).expect("jxl-rs decode failed");
    assert_eq!(decoded_img.width, 32);
    assert_eq!(decoded_img.height, 32);

    // RGB: 3 channels
    let decoded: Vec<u8> = decoded_img
        .pixels
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();
    let mut diff_count = 0;
    let mut max_diff = 0u8;
    for (&orig, &dec) in data.iter().zip(decoded.iter()) {
        let diff = (orig as i16 - dec as i16).unsigned_abs() as u8;
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0 {
            diff_count += 1;
        }
    }
    assert_eq!(
        diff_count, 0,
        "Squeeze RGB 32x32: {} pixels differ, max_diff={}",
        diff_count, max_diff
    );
    eprintln!("Squeeze RGB 32x32: PASS (pixel-exact)");
}

/// Squeeze roundtrip for larger 128x128 gray image.
#[test]
fn test_squeeze_roundtrip_gray_128x128() {
    use crate::{LosslessConfig, PixelLayout};

    let mut data = vec![0u8; 128 * 128];
    for y in 0..128 {
        for x in 0..128 {
            data[y * 128 + x] = ((x * 2 + y) % 256) as u8;
        }
    }

    let bytes = LosslessConfig::new()
        .with_squeeze(true)
        .encode(&data, 128, 128, PixelLayout::Gray8)
        .unwrap();
    eprintln!("Squeeze gray 128x128: {} bytes", bytes.len());

    let decoded_img =
        crate::test_helpers::decode_with_jxl_rs(&bytes).expect("jxl-rs decode failed");
    assert_eq!(decoded_img.width, 128);
    assert_eq!(decoded_img.height, 128);
    let decoded: Vec<u8> = decoded_img
        .pixels
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();
    let mut diff_count = 0;
    for (&orig, &dec) in data.iter().zip(decoded.iter()) {
        if orig != dec {
            diff_count += 1;
        }
    }
    assert_eq!(
        diff_count, 0,
        "Squeeze gray 128x128: {} pixels differ",
        diff_count
    );
    eprintln!("Squeeze gray 128x128: PASS (pixel-exact)");
}

/// Simple encode benchmark for CI — exercises the full lossy pipeline on WASM.
/// Run with: cargo test --release --lib -- bench_encode_256x256 --ignored --nocapture
#[test]
#[ignore]
fn bench_encode_256x256() {
    use crate::{LossyConfig, PixelLayout};

    let (width, height) = (256u32, 256u32);
    let mut data = vec![0u8; (width * height * 3) as usize];
    // Deterministic pseudo-random content via LCG
    let mut seed: u64 = 0xDEAD_BEEF;
    for val in data.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *val = (seed >> 56) as u8;
    }

    // Warmup
    let _ = LossyConfig::new(2.0).encode(&data, width, height, PixelLayout::Rgb8);

    let iters = 3;
    let start = std::time::Instant::now();
    let mut size = 0;
    for _ in 0..iters {
        let encoded = LossyConfig::new(2.0)
            .encode(&data, width, height, PixelLayout::Rgb8)
            .unwrap();
        size = encoded.len();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters;
    let mpixels_per_sec =
        (width as f64 * height as f64 * iters as f64) / elapsed.as_secs_f64() / 1_000_000.0;

    eprintln!(
        "bench_encode_256x256: {per_iter:?}/iter, {mpixels_per_sec:.2} MP/s, {size} bytes output"
    );
}

/// Test squeeze multi-group: 300x300 grayscale (4 groups, simpler channel layout)
#[test]
fn test_squeeze_multigroup_gray_300x300() {
    use crate::{LosslessConfig, PixelLayout};

    let mut data = vec![0u8; 300 * 300];
    for y in 0..300 {
        for x in 0..300 {
            data[y * 300 + x] = ((x * 2 + y) % 256) as u8;
        }
    }

    // Effort 7: exercises squeeze + tree learning + LZ77 without the
    // pathological tree growth that effort 9 causes on synthetic gradients.
    let bytes = LosslessConfig::new()
        .with_squeeze(true)
        .with_effort(7)
        .encode(&data, 300, 300, PixelLayout::Gray8)
        .unwrap();

    eprintln!("Squeeze multi-group gray 300x300: {} bytes", bytes.len());
    crate::test_helpers::save_test_output("squeeze", "squeeze_multigroup_gray_300x300.jxl", &bytes);

    // Decode with jxl-rs
    let decoded_img =
        crate::test_helpers::decode_with_jxl_rs(&bytes).expect("jxl-rs decode failed");
    assert_eq!(decoded_img.width, 300);
    assert_eq!(decoded_img.height, 300);

    let decoded: Vec<u8> = decoded_img
        .pixels
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();

    let mut diff_count = 0;
    for (&orig, &dec) in data.iter().zip(decoded.iter()) {
        if orig != dec {
            diff_count += 1;
        }
    }
    assert_eq!(
        diff_count, 0,
        "Squeeze multi-group gray: {diff_count} pixel differences"
    );
}

/// Test squeeze multi-group: 300x300 RGB (4 groups, simpler)
#[test]
fn test_squeeze_multigroup_rgb_300x300() {
    use crate::{LosslessConfig, PixelLayout};

    let mut data = vec![0u8; 300 * 300 * 3];
    for y in 0..300 {
        for x in 0..300 {
            let i = (y * 300 + x) * 3;
            data[i] = (x % 256) as u8;
            data[i + 1] = (y % 256) as u8;
            data[i + 2] = ((x + y) % 256) as u8;
        }
    }

    // Effort 7: exercises squeeze + tree learning + LZ77 without the
    // pathological tree growth that effort 9 causes on synthetic gradients.
    let bytes = LosslessConfig::new()
        .with_squeeze(true)
        .with_effort(7)
        .encode(&data, 300, 300, PixelLayout::Rgb8)
        .unwrap();

    eprintln!("Squeeze multi-group RGB 300x300: {} bytes", bytes.len());
    crate::test_helpers::save_test_output("squeeze", "squeeze_multigroup_rgb_300x300.jxl", &bytes);

    // Decode with jxl-rs
    let decoded_img =
        crate::test_helpers::decode_with_jxl_rs(&bytes).expect("jxl-rs decode failed");
    assert_eq!(decoded_img.width, 300);
    assert_eq!(decoded_img.height, 300);

    // Check pixel-exact decode
    let decoded: Vec<u8> = decoded_img
        .pixels
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();

    let mut diff_count = 0;
    for (&orig, &dec) in data.iter().zip(decoded.iter()) {
        if orig != dec {
            diff_count += 1;
        }
    }
    assert_eq!(
        diff_count, 0,
        "Squeeze multi-group RGB 300: {diff_count} pixel differences"
    );
}

/// Test squeeze multi-group: 512x512 RGB (4 groups)
#[test]
fn test_squeeze_multigroup_rgb_512x512() {
    use crate::{LosslessConfig, PixelLayout};

    let mut data = vec![0u8; 512 * 512 * 3];
    for y in 0..512 {
        for x in 0..512 {
            let i = (y * 512 + x) * 3;
            data[i] = (x % 256) as u8;
            data[i + 1] = (y % 256) as u8;
            data[i + 2] = ((x + y) % 256) as u8;
        }
    }

    // Effort 7: exercises squeeze + tree learning + LZ77 without the
    // pathological tree growth that effort 9 causes on synthetic gradients
    // (174K unique samples × 256 buckets × 15 properties).
    let bytes = LosslessConfig::new()
        .with_squeeze(true)
        .with_effort(7)
        .encode(&data, 512, 512, PixelLayout::Rgb8)
        .unwrap();

    eprintln!("Squeeze multi-group RGB 512x512: {} bytes", bytes.len());
    crate::test_helpers::save_test_output("squeeze", "squeeze_multigroup_rgb_512x512.jxl", &bytes);

    // Decode with jxl-rs
    let decoded_img =
        crate::test_helpers::decode_with_jxl_rs(&bytes).expect("jxl-rs decode failed");
    assert_eq!(decoded_img.width, 512);
    assert_eq!(decoded_img.height, 512);

    // Check pixel-exact decode
    let decoded: Vec<u8> = decoded_img
        .pixels
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();

    let mut diff_count = 0;
    for (i, (&orig, &dec)) in data.iter().zip(decoded.iter()).enumerate() {
        if orig != dec {
            diff_count += 1;
            if diff_count <= 5 {
                let pixel = i / 3;
                let channel = i % 3;
                eprintln!(
                    "Mismatch at pixel {} ch {}: orig={} decoded={}",
                    pixel, channel, orig, dec
                );
            }
        }
    }
    assert_eq!(
        diff_count, 0,
        "Squeeze multi-group: {diff_count} pixel differences"
    );
}

/// Test RGB lossless encoding at many sizes, checking both jxl-rs and djxl.
/// This test was created to investigate decode failures for certain RGB image sizes.
#[test]
#[ignore]
fn test_rgb_lossless_djxl_sweep() {
    use std::process::Command;

    let mut failures_jxlrs = Vec::new();
    let mut failures_djxl = Vec::new();
    let mut total = 0;

    for w in 4..40 {
        for h in 4..40 {
            total += 1;
            // Hash-based pixels: guaranteed all unique colors -> RCT path (not palette)
            let mut data = vec![0u8; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) * 3;
                    let v = (x as u32)
                        .wrapping_mul(2654435761)
                        .wrapping_add((y as u32).wrapping_mul(2246822519));
                    data[idx] = (v & 0xFF) as u8;
                    data[idx + 1] = ((v >> 8) & 0xFF) as u8;
                    data[idx + 2] = ((v >> 16) & 0xFF) as u8;
                }
            }

            let encoded = LosslessConfig::new()
                .with_tree_learning(false)
                .encode(&data, w as u32, h as u32, PixelLayout::Rgb8)
                .unwrap_or_else(|e| panic!("{}x{}: encoding failed: {}", w, h, e));

            // Test jxl-rs in-process
            match crate::test_helpers::decode_with_jxl_rs(&encoded) {
                Ok(img) => {
                    let decoded: Vec<u8> = img
                        .pixels
                        .iter()
                        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
                        .collect();
                    if decoded != data {
                        failures_jxlrs.push(format!("{}x{} (data mismatch)", w, h));
                    }
                }
                Err(e) => {
                    failures_jxlrs.push(format!("{}x{} ({})", w, h, e));
                }
            }

            // Test djxl
            let path = format!("/tmp/sweep_{}x{}.jxl", w, h);
            std::fs::write(&path, &encoded).unwrap();
            let output = Command::new(crate::test_helpers::djxl_path())
                .args([&path, &format!("/tmp/sweep_{}x{}.png", w, h)])
                .output();
            match output {
                Ok(o) if !o.status.success() => {
                    failures_djxl.push(format!("{}x{}", w, h));
                }
                Err(e) => {
                    failures_djxl.push(format!("{}x{} (launch: {})", w, h, e));
                }
                _ => {}
            }
        }
    }

    eprintln!("\nRGB lossless sweep: {} total sizes tested", total);
    eprintln!("  jxl-rs failures: {} / {}", failures_jxlrs.len(), total);
    for f in &failures_jxlrs[..failures_jxlrs.len().min(20)] {
        eprintln!("    FAIL (jxl-rs): {}", f);
    }
    eprintln!("  djxl failures: {} / {}", failures_djxl.len(), total);
    for f in &failures_djxl[..failures_djxl.len().min(20)] {
        eprintln!("    FAIL (djxl): {}", f);
    }

    // The test itself checks if there are djxl failures
    assert!(
        failures_djxl.is_empty(),
        "djxl decode failures: {}/{} sizes failed. First 10: {:?}",
        failures_djxl.len(),
        total,
        &failures_djxl[..failures_djxl.len().min(10)]
    );
}

/// Test RGB lossless encoding with gradient pattern that previously failed.
/// Pattern: (x*32+y*20+c*80)%256 - produces few unique colors, uses RCT path.
#[test]
#[ignore]
fn test_rgb_lossless_gradient_pattern_sweep() {
    use std::process::Command;

    let mut failures_djxl = Vec::new();
    let mut failures_jxlrs = Vec::new();
    let mut total = 0;

    // Test both with and without tree learning
    for use_tree in [false, true] {
        for w in 4..40 {
            for h in 4..40 {
                total += 1;
                let mut data = vec![0u8; w * h * 3];
                for y in 0..h {
                    for x in 0..w {
                        let idx = (y * w + x) * 3;
                        data[idx] = ((x * 32 + y * 20) % 256) as u8;
                        data[idx + 1] = ((x * 32 + y * 20 + 80) % 256) as u8;
                        data[idx + 2] = ((x * 32 + y * 20 + 160) % 256) as u8;
                    }
                }

                let encoded = LosslessConfig::new()
                    .with_tree_learning(use_tree)
                    .encode(&data, w as u32, h as u32, PixelLayout::Rgb8)
                    .unwrap_or_else(|e| {
                        panic!("{}x{} tree={}: encoding failed: {}", w, h, use_tree, e)
                    });

                // Test jxl-rs in-process
                match crate::test_helpers::decode_with_jxl_rs(&encoded) {
                    Ok(img) => {
                        let decoded: Vec<u8> = img
                            .pixels
                            .iter()
                            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
                            .collect();
                        if decoded != data {
                            failures_jxlrs
                                .push(format!("{}x{} tree={} (data mismatch)", w, h, use_tree));
                        }
                    }
                    Err(e) => {
                        failures_jxlrs.push(format!("{}x{} tree={} ({})", w, h, use_tree, e));
                    }
                }

                // Test djxl
                let tree_str = if use_tree { "tree" } else { "notree" };
                let path = format!("/tmp/grad_{}x{}_{}.jxl", w, h, tree_str);
                std::fs::write(&path, &encoded).unwrap();
                let output = Command::new(crate::test_helpers::djxl_path())
                    .args([&path, &format!("/tmp/grad_{}x{}_{}.png", w, h, tree_str)])
                    .output();
                match output {
                    Ok(o) if !o.status.success() => {
                        failures_djxl.push(format!("{}x{} tree={}", w, h, use_tree));
                    }
                    Err(e) => {
                        failures_djxl
                            .push(format!("{}x{} tree={} (launch: {})", w, h, use_tree, e));
                    }
                    _ => {}
                }
            }
        }
    }

    eprintln!(
        "\nRGB gradient pattern sweep: {} total tests ({} sizes x 2 tree modes)",
        total,
        total / 2
    );
    eprintln!("  jxl-rs failures: {} / {}", failures_jxlrs.len(), total);
    for f in &failures_jxlrs[..failures_jxlrs.len().min(20)] {
        eprintln!("    FAIL (jxl-rs): {}", f);
    }
    eprintln!("  djxl failures: {} / {}", failures_djxl.len(), total);
    for f in &failures_djxl[..failures_djxl.len().min(30)] {
        eprintln!("    FAIL (djxl): {}", f);
    }

    assert!(
        failures_djxl.is_empty(),
        "djxl decode failures: {}/{} tests failed",
        failures_djxl.len(),
        total
    );
}

/// Test tree learning RCT path on various failing cases.
#[test]
#[ignore]
fn test_tree_learning_debug_single() {
    type PixelFn = Box<dyn Fn(usize, usize) -> [u8; 3]>;
    let cases: Vec<(&str, usize, usize, PixelFn)> = vec![
        // Gradient pattern
        (
            "gradient_11x13",
            11,
            13,
            Box::new(|x, y| [((x * 255) / 10) as u8, ((y * 255) / 12) as u8, 128]),
        ),
        // 8-color pattern
        (
            "8colors_16x16",
            16,
            16,
            Box::new(|x, y| {
                let colors: [[u8; 3]; 8] = [
                    [255, 0, 0],
                    [0, 255, 0],
                    [0, 0, 255],
                    [255, 255, 0],
                    [255, 0, 255],
                    [0, 255, 255],
                    [0, 0, 0],
                    [255, 255, 255],
                ];
                colors[(x + y * 3) % 8]
            }),
        ),
        // XY gradient 256x256
        (
            "xy_256x256",
            256,
            256,
            Box::new(|x, y| [x as u8, y as u8, ((x + y) % 256) as u8]),
        ),
    ];

    for (name, w, h, pixel_fn) in &cases {
        let mut data = vec![0u8; w * h * 3];
        for y in 0..*h {
            for x in 0..*w {
                let idx = (y * w + x) * 3;
                let c = pixel_fn(x, y);
                data[idx] = c[0];
                data[idx + 1] = c[1];
                data[idx + 2] = c[2];
            }
        }

        let encoded = LosslessConfig::new()
            .with_tree_learning(true)
            .encode(&data, *w as u32, *h as u32, PixelLayout::Rgb8)
            .unwrap();

        let path = format!("/tmp/tree_debug_{}.jxl", name);
        std::fs::write(&path, &encoded).unwrap();

        // Decode with jxl-rs
        let jxlrs_ok = match crate::test_helpers::decode_with_jxl_rs(&encoded) {
            Ok(img) => {
                let decoded: Vec<u8> = img
                    .pixels
                    .iter()
                    .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
                    .collect();
                let diffs: usize = data
                    .iter()
                    .zip(decoded.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                if diffs > 0 {
                    eprintln!("{}: jxl-rs {diffs} diffs", name);
                }
                diffs == 0
            }
            Err(e) => {
                eprintln!("{}: jxl-rs ERROR: {}", name, e);
                false
            }
        };

        // Decode with djxl and verify pixels
        let djxl_png_path = format!("/tmp/tree_debug_{}.png", name);
        let djxl_status = std::process::Command::new(crate::test_helpers::djxl_path())
            .args([&path, &djxl_png_path])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        let djxl_ok = if djxl_status {
            // Compare decoded pixels
            let decoded_img = image::open(&djxl_png_path).ok();
            if let Some(decoded_img) = decoded_img {
                let decoded_rgb = decoded_img.to_rgb8();
                let decoded_bytes = decoded_rgb.as_raw();
                let diffs: usize = data
                    .iter()
                    .zip(decoded_bytes.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                if diffs > 0 {
                    eprintln!(
                        "{}: djxl {diffs} pixel diffs (of {} total bytes)",
                        name,
                        data.len().min(decoded_bytes.len())
                    );
                    // Print first few diffs
                    let mut printed = 0;
                    for (i, (a, b)) in data.iter().zip(decoded_bytes.iter()).enumerate() {
                        if a != b && printed < 10 {
                            let ch = ["R", "G", "B"][i % 3];
                            let px = i / 3;
                            let py = px / w;
                            let px = px % w;
                            eprintln!("  diff at ({px},{py}) {ch}: expected {a}, got {b}");
                            printed += 1;
                        }
                    }
                }
                diffs == 0
            } else {
                false
            }
        } else {
            false
        };

        eprintln!(
            "{} ({}x{}): {} bytes, jxl-rs={} djxl={}",
            name,
            w,
            h,
            encoded.len(),
            if jxlrs_ok { "PASS" } else { "FAIL" },
            if djxl_ok { "PASS" } else { "FAIL" },
        );
    }
}

/// Minimal progressive test: 64x64, effort 1 (no custom orders, no LZ77, no butteraugli)
#[test]
#[ignore]
fn test_progressive_minimal() {
    use crate::test_helpers::{decode_with_djxl, decode_with_jxl_rs};

    let w = 128;
    let h = 128;
    let mut data = vec![128u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            data[idx] = (x * 2).min(255) as u8;
            data[idx + 1] = (y * 2).min(255) as u8;
            data[idx + 2] = 128;
        }
    }

    // Test across effort levels to exercise different code paths
    // e1-2: DCT8 only; e3: multi-block transforms + custom orders; e5: ANS + pixel-domain loss
    for effort in [1, 3, 5] {
        let config = LossyConfig::new(2.0)
            .with_progressive(ProgressiveMode::QuantizedAcFullAc)
            .with_effort(effort);
        #[cfg(feature = "butteraugli-loop")]
        let config = config.with_butteraugli_iters(0);
        let encoded = config
            .encode(&data, w as u32, h as u32, PixelLayout::Rgb8)
            .unwrap();

        let r = decode_with_jxl_rs(&encoded);
        eprintln!(
            "Progressive e{}: {} bytes, jxl-rs={}",
            effort,
            encoded.len(),
            if r.is_ok() { "OK" } else { "FAIL" }
        );
        assert!(
            r.is_ok(),
            "Progressive e{} jxl-rs decode failed: {:?}",
            effort,
            r.err()
        );
    }

    // Full roundtrip with both decoders at effort 1 (minimal features)
    let encoded = LossyConfig::new(2.0)
        .with_progressive(ProgressiveMode::QuantizedAcFullAc)
        .with_effort(1)
        .encode(&data, w as u32, h as u32, PixelLayout::Rgb8)
        .unwrap();

    let result_djxl = decode_with_djxl(&encoded);
    assert!(result_djxl.is_ok(), "djxl decode failed");
}

/// Test progressive VarDCT encoding (2-pass quantized mode)
#[test]
#[ignore]
fn test_progressive_qprogressive_roundtrip() {
    use crate::test_helpers::{decode_with_djxl, decode_with_jxl_rs};

    // Generate a 128x128 gradient test image (single group, uses multi-section path)
    let w = 128;
    let h = 128;
    let mut data = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            data[idx] = (x * 2).min(255) as u8;
            data[idx + 1] = (y * 2).min(255) as u8;
            data[idx + 2] = (x + y).min(255) as u8;
        }
    }

    // Encode with 2-pass progressive
    let config = LossyConfig::new(1.0)
        .with_progressive(ProgressiveMode::QuantizedAcFullAc)
        .with_effort(5); // lower effort for faster test
    #[cfg(feature = "butteraugli-loop")]
    let config = config.with_butteraugli_iters(0);
    let encoded = config
        .encode(&data, w as u32, h as u32, PixelLayout::Rgb8)
        .unwrap();

    eprintln!(
        "Progressive 2-pass: {} bytes ({} pixels)",
        encoded.len(),
        w * h
    );

    // Save for debugging
    let path = "/tmp/test_progressive_qprog.jxl";
    std::fs::write(path, &encoded).ok();

    // Decode with jxl-rs
    let result = decode_with_jxl_rs(&encoded);
    match &result {
        Ok(decoded) => {
            eprintln!(
                "jxl-rs: OK, {}x{} {} channels",
                decoded.width, decoded.height, decoded.channels
            );
            assert_eq!(decoded.width, w);
            assert_eq!(decoded.height, h);
        }
        Err(e) => {
            eprintln!("jxl-rs: FAILED: {}", e);
        }
    }
    assert!(result.is_ok(), "jxl-rs decode failed");

    // Decode with djxl
    let result_djxl = decode_with_djxl(&encoded);
    match &result_djxl {
        Ok(decoded) => {
            eprintln!(
                "djxl: OK, {}x{} {} channels",
                decoded.width, decoded.height, decoded.channels
            );
            assert_eq!(decoded.width, w);
            assert_eq!(decoded.height, h);
        }
        Err(e) => {
            eprintln!("djxl: FAILED: {}", e);
        }
    }
    assert!(result_djxl.is_ok(), "djxl decode failed");
}

/// Test progressive VarDCT encoding with multi-group image (300x300 = 4 groups)
#[test]
#[ignore]
fn test_progressive_multigroup() {
    use crate::test_helpers::{decode_with_djxl, decode_with_jxl_rs};

    // 300x300 → 2×2 = 4 groups. Use pseudo-random pattern for variety.
    let w = 300;
    let h = 300;
    let mut data = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let v = ((x * 37 + y * 53 + 123) % 256) as u8;
            data[idx] = v;
            data[idx + 1] = v.wrapping_add(80);
            data[idx + 2] = v.wrapping_add(160);
        }
    }

    // Encode without progressive for comparison
    let nonprog_config = LossyConfig::new(2.0).with_effort(5);
    #[cfg(feature = "butteraugli-loop")]
    let nonprog_config = nonprog_config.with_butteraugli_iters(0);
    let nonprog = nonprog_config
        .encode(&data, w as u32, h as u32, PixelLayout::Rgb8)
        .unwrap();

    // Encode with 2-pass progressive, effort 5
    let config = LossyConfig::new(2.0)
        .with_progressive(ProgressiveMode::QuantizedAcFullAc)
        .with_effort(5);
    #[cfg(feature = "butteraugli-loop")]
    let config = config.with_butteraugli_iters(0);
    let encoded = config
        .encode(&data, w as u32, h as u32, PixelLayout::Rgb8)
        .unwrap();

    eprintln!(
        "Non-progressive: {} bytes, Progressive 2-pass: {} bytes ({}x{})",
        nonprog.len(),
        encoded.len(),
        w,
        h
    );
    assert_ne!(
        nonprog.len(),
        encoded.len(),
        "Progressive should produce different file size than non-progressive"
    );

    let path = "/tmp/test_progressive_multigroup.jxl";
    std::fs::write(path, &encoded).ok();

    // Decode with jxl-rs — get detailed error
    let result = decode_with_jxl_rs(&encoded);
    match &result {
        Ok(decoded) => {
            eprintln!(
                "jxl-rs: OK, {}x{} {} channels",
                decoded.width, decoded.height, decoded.channels
            );
        }
        Err(e) => {
            eprintln!("jxl-rs: FAILED: {:?}", e);
        }
    }
    assert!(
        result.is_ok(),
        "jxl-rs decode failed for multi-group progressive"
    );

    // Also test with djxl
    let result_djxl = decode_with_djxl(&encoded);
    match &result_djxl {
        Ok(decoded) => {
            eprintln!(
                "djxl: OK, {}x{} {} channels",
                decoded.width, decoded.height, decoded.channels
            );
        }
        Err(e) => {
            eprintln!("djxl: FAILED: {}", e);
        }
    }
    assert!(
        result_djxl.is_ok(),
        "djxl decode failed for multi-group progressive"
    );
}

/// Test progressive VarDCT encoding with real photo (content-dependent bug)
#[test]
#[ignore]
fn test_progressive_multigroup_photo() {
    use crate::test_helpers::{decode_with_djxl, decode_with_jxl_rs};

    // Load a real CLIC photo crop (300x300 from 1024x1024)
    let path = "/tmp/test300.png";
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping: {} not found", path);
        return;
    }
    let img = image::open(path).unwrap().to_rgb8();
    let w = img.width() as usize;
    let h = img.height() as usize;
    let data = img.as_raw().as_slice();
    eprintln!("Photo: {}x{}, {} bytes", w, h, data.len());

    // Encode with 2-pass progressive
    let config = LossyConfig::new(2.0)
        .with_progressive(ProgressiveMode::QuantizedAcFullAc)
        .with_effort(5);
    #[cfg(feature = "butteraugli-loop")]
    let config = config.with_butteraugli_iters(0);
    let encoded = config
        .encode(data, w as u32, h as u32, PixelLayout::Rgb8)
        .unwrap();

    eprintln!("Progressive 2-pass photo: {} bytes", encoded.len());
    std::fs::write("/tmp/test_progressive_photo.jxl", &encoded).ok();

    // Decode with jxl-rs
    let result = decode_with_jxl_rs(&encoded);
    match &result {
        Ok(decoded) => {
            eprintln!("jxl-rs: OK, {}x{}", decoded.width, decoded.height);
        }
        Err(e) => {
            eprintln!("jxl-rs: FAILED: {:?}", e);
        }
    }

    // Decode with djxl
    let result_djxl = decode_with_djxl(&encoded);
    match &result_djxl {
        Ok(decoded) => {
            eprintln!("djxl: OK, {}x{}", decoded.width, decoded.height);
        }
        Err(e) => {
            eprintln!("djxl: FAILED: {}", e);
        }
    }

    assert!(result.is_ok(), "jxl-rs decode failed for photo progressive");
    assert!(
        result_djxl.is_ok(),
        "djxl decode failed for photo progressive"
    );
}

/// Test progressive VarDCT encoding (3-pass DC/VLF/LF/AC mode)
#[test]
#[ignore]
fn test_progressive_3pass_roundtrip() {
    use crate::test_helpers::{decode_with_djxl, decode_with_jxl_rs};

    // 128x128 gradient
    let w = 128;
    let h = 128;
    let mut data = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            data[idx] = (x * 2).min(255) as u8;
            data[idx + 1] = (y * 2).min(255) as u8;
            data[idx + 2] = (x + y).min(255) as u8;
        }
    }

    // Encode with 3-pass progressive
    let config = LossyConfig::new(1.0)
        .with_progressive(ProgressiveMode::DcVlfLfAc)
        .with_effort(5);
    #[cfg(feature = "butteraugli-loop")]
    let config = config.with_butteraugli_iters(0);
    let encoded = config
        .encode(&data, w as u32, h as u32, PixelLayout::Rgb8)
        .unwrap();

    eprintln!(
        "Progressive 3-pass: {} bytes ({} pixels)",
        encoded.len(),
        w * h
    );

    let path = "/tmp/test_progressive_3pass.jxl";
    std::fs::write(path, &encoded).ok();

    // Decode with jxl-rs
    let result = decode_with_jxl_rs(&encoded);
    match &result {
        Ok(decoded) => {
            eprintln!(
                "jxl-rs: OK, {}x{} {} channels",
                decoded.width, decoded.height, decoded.channels
            );
        }
        Err(e) => {
            eprintln!("jxl-rs: FAILED: {}", e);
        }
    }
    assert!(result.is_ok(), "jxl-rs decode failed");

    // Decode with djxl
    let result_djxl = decode_with_djxl(&encoded);
    match &result_djxl {
        Ok(decoded) => {
            eprintln!(
                "djxl: OK, {}x{} {} channels",
                decoded.width, decoded.height, decoded.channels
            );
        }
        Err(e) => {
            eprintln!("djxl: FAILED: {}", e);
        }
    }
    assert!(result_djxl.is_ok(), "djxl decode failed");
}

// ── Spline roundtrip tests ──────────────────────────────────────────────────

/// Test that encoding with a simple diagonal spline produces a valid JXL that
/// jxl-rs can decode.
#[test]
fn test_splines_roundtrip_jxl_rs() {
    use crate::test_helpers::decode_with_jxl_rs;
    use crate::{LossyConfig, PixelLayout, Spline, SplinePoint};

    let (width, height) = (128, 128);
    // Medium-gray background so the spline is visible.
    let data = vec![128u8; width * height * 3];

    // One diagonal spline with Y luminance, small sigma.
    let spline = Spline {
        control_points: vec![
            SplinePoint::new(10.0, 10.0),
            SplinePoint::new(60.0, 60.0),
            SplinePoint::new(110.0, 110.0),
        ],
        color_dct: {
            let mut dct = [[0.0f32; 32]; 3];
            dct[1][0] = 0.3; // Y DC — visible luminance
            dct
        },
        sigma_dct: {
            let mut s = [0.0f32; 32];
            s[0] = 1.5; // sigma DC — narrow line
            s
        },
    };

    let encoded = LossyConfig::new(1.0)
        .with_splines(vec![spline])
        .encode(&data, width as u32, height as u32, PixelLayout::Rgb8)
        .expect("encoding with splines failed");

    // Save for manual inspection.
    crate::test_helpers::save_test_output("splines", "diagonal_d1.jxl", &encoded);

    // Decode with jxl-rs — this is the primary validation.
    let decoded = decode_with_jxl_rs(&encoded).expect("jxl-rs failed to decode spline JXL");
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    eprintln!(
        "test_splines_roundtrip_jxl_rs: PASSED ({} bytes, {}x{} decoded)",
        encoded.len(),
        decoded.width,
        decoded.height,
    );
}

/// Test that encoding with splines produces a valid JXL that djxl can decode.
#[test]
fn test_splines_roundtrip_djxl() {
    crate::skip_without_binary!(crate::test_helpers::djxl_path());
    use crate::test_helpers::decode_with_djxl;
    use crate::{LossyConfig, PixelLayout, Spline, SplinePoint};

    let (width, height) = (128, 128);
    let data = vec![128u8; width * height * 3];

    let spline = Spline {
        control_points: vec![
            SplinePoint::new(10.0, 10.0),
            SplinePoint::new(60.0, 60.0),
            SplinePoint::new(110.0, 110.0),
        ],
        color_dct: {
            let mut dct = [[0.0f32; 32]; 3];
            dct[1][0] = 0.3;
            dct
        },
        sigma_dct: {
            let mut s = [0.0f32; 32];
            s[0] = 1.5;
            s
        },
    };

    let encoded = LossyConfig::new(1.0)
        .with_splines(vec![spline])
        .encode(&data, width as u32, height as u32, PixelLayout::Rgb8)
        .expect("encoding with splines failed");

    crate::test_helpers::save_test_output("splines", "diagonal_d1_djxl.jxl", &encoded);

    let decoded = decode_with_djxl(&encoded).expect("djxl failed to decode spline JXL");
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    eprintln!(
        "test_splines_roundtrip_djxl: PASSED ({} bytes, {}x{} decoded)",
        encoded.len(),
        decoded.width,
        decoded.height,
    );
}

/// Test that encoding WITHOUT splines produces identical output to baseline.
/// Ensures the spline plumbing doesn't regress anything when unused.
#[test]
fn test_no_splines_baseline() {
    use crate::{LossyConfig, PixelLayout};

    let (width, height) = (32, 32);
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = (x * 8) as u8;
            data[idx + 1] = (y * 8) as u8;
            data[idx + 2] = 128;
        }
    }

    // Encode without splines (normal path).
    let encoded_no_splines = LossyConfig::new(1.0)
        .encode(&data, width as u32, height as u32, PixelLayout::Rgb8)
        .expect("encoding without splines failed");

    // Encode with empty splines vec — should be identical to no-splines.
    let encoded_empty = LossyConfig::new(1.0)
        .with_splines(vec![])
        .encode(&data, width as u32, height as u32, PixelLayout::Rgb8)
        .expect("encoding with empty splines failed");

    assert_eq!(
        encoded_no_splines.len(),
        encoded_empty.len(),
        "empty splines should produce identical output to no-splines"
    );
    assert_eq!(
        encoded_no_splines, encoded_empty,
        "empty splines should produce byte-identical output"
    );
}

/// Test lossy VarDCT encoding of grayscale input (Gray8).
/// Validates that the file header correctly signals ColorSpace::Gray
/// and that jxl-rs can decode the output.
#[test]
fn test_lossy_grayscale_roundtrip_jxl_rs() {
    use crate::test_helpers::decode_with_jxl_rs;
    use crate::{LossyConfig, PixelLayout};

    // 32x32 grayscale gradient
    let mut data = vec![0u8; 32 * 32];
    for y in 0..32 {
        for x in 0..32 {
            data[y * 32 + x] = ((x * 8 + y * 4) % 256) as u8;
        }
    }

    let encoded = LossyConfig::new(1.0)
        .with_effort(5)
        .encode(&data, 32, 32, PixelLayout::Gray8)
        .expect("grayscale lossy encoding failed");

    // Verify JXL signature
    assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);

    // Decode with jxl-rs
    let decoded = decode_with_jxl_rs(&encoded).expect("jxl-rs failed to decode grayscale lossy");
    assert_eq!(decoded.width, 32);
    assert_eq!(decoded.height, 32);
    // Grayscale VarDCT output should be 1 channel (decoder converts XYB → gray)
    assert_eq!(
        decoded.channels, 1,
        "expected 1 channel for grayscale output"
    );

    // Verify lossy quality: pixels should be somewhat close to originals
    let mut max_diff = 0.0f32;
    for y in 0..32 {
        for x in 0..32 {
            let original = data[y * 32 + x] as f32 / 255.0;
            let decoded_val = decoded.get(x, y, 0);
            let diff = (original - decoded_val).abs();
            max_diff = max_diff.max(diff);
        }
    }
    eprintln!("Grayscale lossy d=1.0: max pixel diff = {:.4}", max_diff);
    assert!(
        max_diff < 0.15,
        "max pixel diff {:.4} too high for d=1.0",
        max_diff
    );
}

/// Test lossy VarDCT encoding of grayscale+alpha input (GrayAlpha8).
#[test]
fn test_lossy_grayscale_alpha_roundtrip_jxl_rs() {
    use crate::test_helpers::decode_with_jxl_rs;
    use crate::{LossyConfig, PixelLayout};

    // 16x16 grayscale+alpha
    let mut data = vec![0u8; 16 * 16 * 2];
    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 2;
            data[idx] = ((x * 16 + y * 8) % 256) as u8; // gray
            data[idx + 1] = 255; // alpha = opaque
        }
    }

    let encoded = LossyConfig::new(1.0)
        .with_effort(5)
        .encode(&data, 16, 16, PixelLayout::GrayAlpha8)
        .expect("grayscale+alpha lossy encoding failed");

    assert_eq!(&encoded[0..2], &[0xFF, 0x0A]);

    let decoded =
        decode_with_jxl_rs(&encoded).expect("jxl-rs failed to decode grayscale+alpha lossy");
    assert_eq!(decoded.width, 16);
    assert_eq!(decoded.height, 16);
    // Should be 2 channels: gray + alpha
    assert_eq!(
        decoded.channels, 2,
        "expected 2 channels for grayscale+alpha output"
    );
}

/// Test lossy grayscale decodes with djxl (libjxl reference decoder).
#[test]
fn test_lossy_grayscale_roundtrip_djxl() {
    crate::skip_without_binary!(crate::test_helpers::djxl_path());
    use crate::test_helpers::decode_with_djxl;
    use crate::{LossyConfig, PixelLayout};

    // 32x32 grayscale gradient
    let mut data = vec![0u8; 32 * 32];
    for y in 0..32 {
        for x in 0..32 {
            data[y * 32 + x] = ((x * 8 + y * 4) % 256) as u8;
        }
    }

    let encoded = LossyConfig::new(1.0)
        .with_effort(5)
        .encode(&data, 32, 32, PixelLayout::Gray8)
        .expect("grayscale lossy encoding failed");

    let decoded = decode_with_djxl(&encoded).expect("djxl failed to decode grayscale lossy");
    assert_eq!(decoded.width, 32);
    assert_eq!(decoded.height, 32);
}

// ── Streaming encoder tests ─────────────────────────────────────────────────

#[test]
fn test_streaming_lossy_matches_oneshot() {
    // Encode a small image both ways and verify identical output.
    let w = 16u32;
    let h = 16;
    let pixels: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();

    let cfg = LossyConfig::new(2.0).with_effort(3);

    let oneshot = cfg.encode(&pixels, w, h, PixelLayout::Rgb8).unwrap();

    let mut enc = cfg.encoder(w, h, PixelLayout::Rgb8).unwrap();
    enc.push_rows(&pixels, h).unwrap();
    let streaming = enc.finish().unwrap();

    assert_eq!(oneshot, streaming, "streaming and oneshot output differ");
}

#[test]
fn test_streaming_lossy_row_at_a_time() {
    let w = 8u32;
    let h = 8;
    let pixels: Vec<u8> = (0..w * h * 3).map(|i| (i % 199) as u8).collect();
    let row_bytes = w as usize * 3;

    let cfg = LossyConfig::new(2.0).with_effort(3);

    let oneshot = cfg.encode(&pixels, w, h, PixelLayout::Rgb8).unwrap();

    let mut enc = cfg.encoder(w, h, PixelLayout::Rgb8).unwrap();
    for row in 0..h {
        let start = row as usize * row_bytes;
        enc.push_rows(&pixels[start..start + row_bytes], 1).unwrap();
    }
    assert_eq!(enc.rows_pushed(), h);
    let streaming = enc.finish().unwrap();

    assert_eq!(
        oneshot, streaming,
        "row-at-a-time streaming differs from oneshot"
    );
}

#[test]
fn test_streaming_lossy_rgba() {
    let w = 8u32;
    let h = 8;
    let pixels: Vec<u8> = (0..w * h * 4).map(|i| (i % 211) as u8).collect();

    let cfg = LossyConfig::new(2.0).with_effort(3);

    let oneshot = cfg.encode(&pixels, w, h, PixelLayout::Rgba8).unwrap();

    let mut enc = cfg.encoder(w, h, PixelLayout::Rgba8).unwrap();
    // Push in two halves
    let half = h / 2;
    let mid = half as usize * w as usize * 4;
    enc.push_rows(&pixels[..mid], half).unwrap();
    enc.push_rows(&pixels[mid..], h - half).unwrap();
    let streaming = enc.finish().unwrap();

    assert_eq!(oneshot, streaming);
}

#[test]
fn test_streaming_lossless_matches_oneshot() {
    let w = 16u32;
    let h = 16;
    let pixels: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();

    let cfg = LosslessConfig::new().with_effort(3);

    let oneshot = cfg.encode(&pixels, w, h, PixelLayout::Rgb8).unwrap();

    let mut enc = cfg.encoder(w, h, PixelLayout::Rgb8).unwrap();
    enc.push_rows(&pixels, h).unwrap();
    let streaming = enc.finish().unwrap();

    assert_eq!(oneshot, streaming, "lossless streaming and oneshot differ");
}

#[test]
fn test_streaming_lossless_row_at_a_time() {
    let w = 8u32;
    let h = 8;
    let pixels: Vec<u8> = (0..w * h * 3).map(|i| (i % 199) as u8).collect();
    let row_bytes = w as usize * 3;

    let cfg = LosslessConfig::new().with_effort(3);
    let oneshot = cfg.encode(&pixels, w, h, PixelLayout::Rgb8).unwrap();

    let mut enc = cfg.encoder(w, h, PixelLayout::Rgb8).unwrap();
    for row in 0..h {
        let start = row as usize * row_bytes;
        enc.push_rows(&pixels[start..start + row_bytes], 1).unwrap();
    }
    let streaming = enc.finish().unwrap();

    assert_eq!(
        oneshot, streaming,
        "lossless row-at-a-time differs from oneshot"
    );
}

#[test]
fn test_streaming_lossless_gray() {
    let w = 8u32;
    let h = 8;
    let pixels: Vec<u8> = (0..w * h).map(|i| (i % 200) as u8).collect();

    let cfg = LosslessConfig::new().with_effort(3);
    let oneshot = cfg.encode(&pixels, w, h, PixelLayout::Gray8).unwrap();

    let mut enc = cfg.encoder(w, h, PixelLayout::Gray8).unwrap();
    enc.push_rows(&pixels, h).unwrap();
    let streaming = enc.finish().unwrap();

    assert_eq!(oneshot, streaming);
}

#[test]
fn test_streaming_lossless_rgba() {
    let w = 8u32;
    let h = 8;
    let pixels: Vec<u8> = (0..w * h * 4).map(|i| (i % 211) as u8).collect();

    let cfg = LosslessConfig::new().with_effort(3);
    let oneshot = cfg.encode(&pixels, w, h, PixelLayout::Rgba8).unwrap();

    let mut enc = cfg.encoder(w, h, PixelLayout::Rgba8).unwrap();
    enc.push_rows(&pixels[..w as usize * 4 * 4], 4).unwrap();
    enc.push_rows(&pixels[w as usize * 4 * 4..], 4).unwrap();
    let streaming = enc.finish().unwrap();

    assert_eq!(oneshot, streaming);
}

#[test]
fn test_streaming_error_too_many_rows() {
    let w = 4u32;
    let h = 4;
    let pixels = vec![0u8; w as usize * h as usize * 3];

    let mut enc = LossyConfig::new(2.0)
        .encoder(w, h, PixelLayout::Rgb8)
        .unwrap();
    // Push all rows, then try to push more
    enc.push_rows(&pixels, h).unwrap();
    let err = enc.push_rows(&[0u8; 12], 1);
    assert!(err.is_err(), "should reject rows beyond height");
}

#[test]
fn test_streaming_error_incomplete_finish() {
    let w = 4u32;
    let h = 4;

    let mut enc = LosslessConfig::new()
        .encoder(w, h, PixelLayout::Rgb8)
        .unwrap();
    // Push only 2 of 4 rows
    enc.push_rows(&vec![0u8; w as usize * 2 * 3], 2).unwrap();
    let err = enc.finish();
    assert!(err.is_err(), "should reject incomplete image on finish");
}

#[test]
fn test_streaming_error_wrong_buffer_size() {
    let w = 4u32;
    let h = 4;

    let mut enc = LossyConfig::new(2.0)
        .encoder(w, h, PixelLayout::Rgb8)
        .unwrap();
    // Pass wrong number of bytes
    let err = enc.push_rows(&[0u8; 10], 1);
    assert!(err.is_err(), "should reject wrong buffer size");
}

#[test]
fn test_streaming_zero_rows_noop() {
    let w = 4u32;
    let h = 4;

    let mut enc = LossyConfig::new(2.0)
        .encoder(w, h, PixelLayout::Rgb8)
        .unwrap();
    // Pushing zero rows should be a no-op
    enc.push_rows(&[], 0).unwrap();
    assert_eq!(enc.rows_pushed(), 0);
}

#[test]
fn test_streaming_lossy_finish_into() {
    let w = 8u32;
    let h = 8;
    let pixels: Vec<u8> = (0..w * h * 3).map(|i| (i % 199) as u8).collect();

    let cfg = LossyConfig::new(2.0).with_effort(3);
    let oneshot = cfg.encode(&pixels, w, h, PixelLayout::Rgb8).unwrap();

    let mut enc = cfg.encoder(w, h, PixelLayout::Rgb8).unwrap();
    enc.push_rows(&pixels, h).unwrap();
    let mut out = Vec::new();
    let result = enc.finish_into(&mut out).unwrap();
    assert_eq!(out, oneshot);
    assert!(result.stats().codestream_size() > 0);
}

#[test]
fn test_streaming_lossy_16bit() {
    let w = 8u32;
    let h = 8;
    let pixels_u16: Vec<u16> = (0..w * h * 3).map(|i| (i * 100 % 65535) as u16).collect();
    let pixels: &[u8] = bytemuck::cast_slice(&pixels_u16);

    let cfg = LossyConfig::new(2.0).with_effort(3);
    let oneshot = cfg.encode(pixels, w, h, PixelLayout::Rgb16).unwrap();

    let mut enc = cfg.encoder(w, h, PixelLayout::Rgb16).unwrap();
    enc.push_rows(pixels, h).unwrap();
    let streaming = enc.finish().unwrap();

    assert_eq!(oneshot, streaming);
}

#[test]
fn test_streaming_lossless_16bit() {
    let w = 8u32;
    let h = 8;
    let pixels_u16: Vec<u16> = (0..w * h * 3).map(|i| (i * 100 % 65535) as u16).collect();
    let pixels: &[u8] = bytemuck::cast_slice(&pixels_u16);

    let cfg = LosslessConfig::new().with_effort(3);
    let oneshot = cfg.encode(pixels, w, h, PixelLayout::Rgb16).unwrap();

    let mut enc = cfg.encoder(w, h, PixelLayout::Rgb16).unwrap();
    enc.push_rows(pixels, h).unwrap();
    let streaming = enc.finish().unwrap();

    assert_eq!(oneshot, streaming);
}
