// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Test helpers to prevent false positives and verify what tests actually do.
//!
//! IMPORTANT: Use jxl-rs as the PRIMARY decoder for all roundtrip tests.
//! jxl-oxide is only a secondary/fallback decoder.

use crate::error::Result;

/// Encoding mode in JXL
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingMode {
    VarDct,  // encoding=0 (lossy)
    Modular, // encoding=1 (lossless)
}

/// Decoded image result from jxl-rs
pub struct DecodedImage {
    /// Width in pixels
    pub width: usize,
    /// Height in pixels
    pub height: usize,
    /// Number of color channels (3 for RGB, 4 for RGBA)
    pub channels: usize,
    /// Pixel data as interleaved f32 values [R, G, B, ...] row by row
    pub pixels: Vec<f32>,
}

impl DecodedImage {
    /// Get pixel value at (x, y) for channel c
    pub fn get(&self, x: usize, y: usize, c: usize) -> f32 {
        let idx = (y * self.width + x) * self.channels + c;
        self.pixels[idx]
    }

    /// Get RGB pixel as (r, g, b) scaled to 0-255
    pub fn get_rgb_u8(&self, x: usize, y: usize) -> (u8, u8, u8) {
        let r = (self.get(x, y, 0) * 255.0).clamp(0.0, 255.0) as u8;
        let g = (self.get(x, y, 1) * 255.0).clamp(0.0, 255.0) as u8;
        let b = (self.get(x, y, 2) * 255.0).clamp(0.0, 255.0) as u8;
        (r, g, b)
    }
}

/// Decode JXL data using jxl-rs (PRIMARY decoder for single-group images).
///
/// WARNING: jxl-rs has a multi-group VarDCT decoder bug (same as jxl-oxide).
/// For images >256x256, use decode_with_djxl() instead.
///
/// Returns decoded image with f32 pixel values.
pub fn decode_with_jxl_rs(data: &[u8]) -> Result<DecodedImage> {
    use jxl::api::states::Initialized;
    use jxl::api::{
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult,
    };
    use jxl::image::{Image, Rect};

    let options = JxlDecoderOptions::default();
    let mut decoder: JxlDecoder<Initialized> = JxlDecoder::new(options);
    let mut input = data;

    // Process until we have image info
    let mut decoder = loop {
        match decoder
            .process(&mut input)
            .map_err(|e| crate::error::Error::InvalidInput(format!("jxl-rs init error: {:?}", e)))?
        {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err(crate::error::Error::InvalidInput(
                        "jxl-rs: unexpected end of input during header parsing".to_string(),
                    ));
                }
                decoder = fallback;
            }
        }
    };

    // Get basic info
    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;

    // Request f32 RGB format
    let default_format = decoder.current_pixel_format();
    let num_channels = default_format.color_type.samples_per_pixel();

    let requested_format = JxlPixelFormat {
        color_type: default_format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: default_format
            .extra_channel_format
            .iter()
            .map(|_| Some(JxlDataFormat::f32()))
            .collect(),
    };
    decoder.set_pixel_format(requested_format);

    // Process until we have frame info
    let mut decoder = loop {
        match decoder.process(&mut input).map_err(|e| {
            crate::error::Error::InvalidInput(format!("jxl-rs frame error: {:?}", e))
        })? {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err(crate::error::Error::InvalidInput(
                        "jxl-rs: unexpected end of input during frame parsing".to_string(),
                    ));
                }
                decoder = fallback;
            }
        }
    };

    // Create output buffer
    let mut color_buffer = Image::<f32>::new((width * num_channels, height)).map_err(|e| {
        crate::error::Error::InvalidInput(format!("jxl-rs buffer alloc error: {:?}", e))
    })?;

    let mut buffers: Vec<_> = vec![JxlOutputBuffer::from_image_rect_mut(
        color_buffer
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * num_channels, height),
            })
            .into_raw(),
    )];

    // Decode frame
    loop {
        match decoder.process(&mut input, &mut buffers).map_err(|e| {
            crate::error::Error::InvalidInput(format!("jxl-rs decode error: {:?}", e))
        })? {
            ProcessingResult::Complete { .. } => break,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err(crate::error::Error::InvalidInput(
                        "jxl-rs: unexpected end of input during decode".to_string(),
                    ));
                }
                decoder = fallback;
            }
        }
    }

    // Extract pixels from buffer
    let mut pixels = Vec::with_capacity(width * height * num_channels);
    for y in 0..height {
        let row = color_buffer.row(y);
        pixels.extend_from_slice(row);
    }

    Ok(DecodedImage {
        width,
        height,
        channels: num_channels,
        pixels,
    })
}

/// Decode JXL data using djxl (libjxl reference decoder).
///
/// This is the GOLD STANDARD decoder. Use for multi-group VarDCT images
/// since both jxl-rs and jxl-oxide have multi-group decoder bugs.
///
/// Requires djxl binary at `/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl`
pub fn decode_with_djxl(data: &[u8]) -> Result<DecodedImage> {
    use std::process::Command;

    // Use unique temp file names to avoid race conditions
    let pid = std::process::id();
    let temp_jxl = format!("/tmp/decode_test_djxl_{}.jxl", pid);
    let temp_png = format!("/tmp/decode_test_djxl_{}.png", pid);

    std::fs::write(&temp_jxl, data).map_err(|e| {
        crate::error::Error::InvalidInput(format!("Failed to write temp file: {:?}", e))
    })?;

    // Run djxl
    let djxl_path = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";
    let output = Command::new(djxl_path)
        .args([&temp_jxl, &temp_png])
        .output()
        .map_err(|e| crate::error::Error::InvalidInput(format!("Failed to run djxl: {:?}", e)))?;

    if !output.status.success() {
        let _ = std::fs::remove_file(&temp_jxl);
        return Err(crate::error::Error::InvalidInput(format!(
            "djxl failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    // Load PNG with image crate
    let img = image::open(&temp_png).map_err(|e| {
        let _ = std::fs::remove_file(&temp_jxl);
        let _ = std::fs::remove_file(&temp_png);
        crate::error::Error::InvalidInput(format!("Failed to load decoded PNG: {:?}", e))
    })?;
    let rgb = img.to_rgb8();

    let width = rgb.width() as usize;
    let height = rgb.height() as usize;

    // Convert u8 to f32
    let pixels: Vec<f32> = rgb.as_raw().iter().map(|&v| v as f32 / 255.0).collect();

    // Debug: check the actual pixel values we're returning
    eprintln!(
        "DEBUG decode_with_djxl: {}x{}, first 9 u8 raw: {:?}",
        width,
        height,
        rgb.as_raw().iter().take(9).copied().collect::<Vec<_>>()
    );

    // Cleanup temp files
    let _ = std::fs::remove_file(&temp_jxl);
    let _ = std::fs::remove_file(&temp_png);

    Ok(DecodedImage {
        width,
        height,
        channels: 3,
        pixels,
    })
}

/// Decode JXL data using jxl-oxide (SECONDARY decoder).
///
/// WARNING: jxl-oxide has a multi-group VarDCT decoder bug.
/// For images >256x256, use decode_with_djxl() instead.
pub fn decode_with_jxl_oxide(data: &[u8]) -> Result<DecodedImage> {
    let mut image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .map_err(|e| {
            crate::error::Error::InvalidInput(format!("jxl-oxide decode failed: {:?}", e))
        })?;

    // Request linear sRGB output so decoded pixels are in linear RGB space,
    // matching our encoder's internal representation.
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));

    let width = image.width() as usize;
    let height = image.height() as usize;
    let channels = image.pixel_format().channels();

    // Render to get actual pixels (linear f32)
    let render = image.render_frame(0).map_err(|e| {
        crate::error::Error::InvalidInput(format!("jxl-oxide render failed: {:?}", e))
    })?;

    // Get pixel data as f32 (interleaved)
    let framebuffer = render.image_all_channels();
    let buf = framebuffer.buf();

    // buf is already interleaved [R,G,B,R,G,B,...] for all pixels
    let pixels = buf.to_vec();

    Ok(DecodedImage {
        width,
        height,
        channels,
        pixels,
    })
}

/// Parse the encoding mode from a JXL bitstream.
/// Returns None if unable to parse (ambiguous or invalid).
pub fn parse_encoding_mode(data: &[u8]) -> Option<EncodingMode> {
    if data.len() < 10 {
        return None;
    }

    // Read bit at position (LSB-first)
    fn read_bit(data: &[u8], bit_pos: usize) -> Option<u8> {
        let byte_idx = bit_pos / 8;
        let bit_idx = bit_pos % 8;
        if byte_idx >= data.len() {
            return None;
        }
        Some((data[byte_idx] >> bit_idx) & 1)
    }

    // Frame header typically starts around bit 38-60 depending on size header
    // Look for all_default=0 followed by frame_type (2 bits) and encoding (1 bit)
    // Start at 38 to skip file header metadata (which can have spurious zeros)
    // The frame header position varies by file header size and the bit parsing
    // is fragile. Since VarDctEncoder always produces VarDCT (verified in source)
    // and the real test is that decoders work, we use a simpler heuristic:
    // Just check if the file decodes and trust the encoding type based on API used.
    //
    // For robustness, search byte-aligned positions for the frame header pattern.
    for start_byte in 4..25 {
        let start_bit = start_byte * 8;
        let all_default = read_bit(data, start_bit)?;
        if all_default == 0 {
            // all_default=0, so frame_type (2 bits) and encoding (1 bit) follow
            let frame_type_0 = read_bit(data, start_bit + 1)?;
            let frame_type_1 = read_bit(data, start_bit + 2)?;
            let encoding_bit = read_bit(data, start_bit + 3)?;
            // For a valid frame: frame_type should be 0 (regular frame)
            if frame_type_0 == 0 && frame_type_1 == 0 {
                return Some(match encoding_bit {
                    0 => EncodingMode::VarDct,
                    1 => EncodingMode::Modular,
                    _ => unreachable!(),
                });
            }
        }
    }

    None
}

/// Assert that encoded data uses the expected encoding mode.
/// Panics with a clear message if the mode doesn't match.
pub fn assert_encoding_mode(data: &[u8], expected: EncodingMode, test_name: &str) {
    let actual = parse_encoding_mode(data).unwrap_or_else(|| {
        panic!(
            "{}: Could not parse encoding mode from bitstream",
            test_name
        )
    });

    assert_eq!(
        actual, expected,
        "{}: Expected {:?} but got {:?}. This test is not testing what it claims!",
        test_name, expected, actual
    );
}

/// Standard roundtrip test for lossless encoding.
/// Encodes with Modular, verifies encoding mode, then decodes with jxl-rs (primary).
pub fn test_lossless_roundtrip(
    data: &[u8],
    width: usize,
    height: usize,
    test_name: &str,
) -> Result<()> {
    let encoded = crate::LosslessConfig::new()
        .encode(data, width as u32, height as u32, crate::PixelLayout::Rgb8)
        .map_err(|e| crate::error::Error::InvalidInput(format!("{e}")))?;

    // VERIFY we actually used Modular encoding
    assert_encoding_mode(&encoded, EncodingMode::Modular, test_name);

    // Decode with jxl-rs (PRIMARY decoder)
    let decoded = decode_with_jxl_rs(&encoded)?;
    assert_eq!(decoded.width, width, "{}: width mismatch", test_name);
    assert_eq!(decoded.height, height, "{}: height mismatch", test_name);

    Ok(())
}

/// Standard roundtrip test for lossy VarDCT encoding.
/// Encodes with VarDCT, verifies encoding mode, then decodes with jxl-rs (primary).
pub fn test_lossy_roundtrip(
    data: &[u8],
    width: usize,
    height: usize,
    distance: f32,
    test_name: &str,
) -> Result<()> {
    let encoded = crate::LossyConfig::new(distance)
        .encode(data, width as u32, height as u32, crate::PixelLayout::Rgb8)
        .map_err(|e| crate::error::Error::InvalidInput(format!("{e}")))?;

    // Save for debugging
    let debug_path = format!("/tmp/{}.jxl", test_name);
    std::fs::write(&debug_path, &encoded).ok();
    eprintln!("DEBUG: Saved {} bytes to {}", encoded.len(), debug_path);

    // VERIFY we actually used VarDCT encoding
    assert_encoding_mode(&encoded, EncodingMode::VarDct, test_name);

    // Decode with jxl-rs (PRIMARY decoder)
    eprintln!("DEBUG: Decoding with jxl-rs (primary)...");
    let decoded = decode_with_jxl_rs(&encoded)?;
    assert_eq!(decoded.width, width, "{}: width mismatch", test_name);
    assert_eq!(decoded.height, height, "{}: height mismatch", test_name);

    Ok(())
}

/// Lossy roundtrip test with quality verification using SSIMULACRA2.
/// Returns SSIM2 score (higher is better, typically 50+ is acceptable).
pub fn test_lossy_roundtrip_with_quality(
    data: &[u8],
    width: usize,
    height: usize,
    distance: f32,
    test_name: &str,
) -> Result<f64> {
    let encoded = crate::LossyConfig::new(distance)
        .encode(data, width as u32, height as u32, crate::PixelLayout::Rgb8)
        .map_err(|e| crate::error::Error::InvalidInput(format!("{e}")))?;

    // Save for debugging
    let debug_path = format!("/tmp/{}.jxl", test_name);
    std::fs::write(&debug_path, &encoded).ok();

    // VERIFY we actually used VarDCT encoding
    assert_encoding_mode(&encoded, EncodingMode::VarDct, test_name);

    // Decode with jxl-rs (PRIMARY decoder)
    let decoded = decode_with_jxl_rs(&encoded)?;
    assert_eq!(decoded.width, width, "{}: width mismatch", test_name);
    assert_eq!(decoded.height, height, "{}: height mismatch", test_name);

    // Calculate SSIMULACRA2 score
    let ssim2 = calculate_ssim2(data, &decoded, width, height);

    eprintln!(
        "{}: encoded {} bytes, SSIM2={:.2}",
        test_name,
        encoded.len(),
        ssim2
    );

    Ok(ssim2)
}

/// Calculate SSIMULACRA2 score between original RGB8 data and decoded image.
/// Returns score where 100 = identical, 90+ = imperceptible, <50 = significant degradation.
pub fn calculate_ssim2(
    original: &[u8],
    decoded: &DecodedImage,
    width: usize,
    height: usize,
) -> f64 {
    use fast_ssim2::compute_ssimulacra2;
    use imgref::ImgVec;

    // Convert original to [u8; 3] array format
    let original_rgb: Vec<[u8; 3]> = original
        .chunks_exact(3)
        .map(|rgb| [rgb[0], rgb[1], rgb[2]])
        .collect();

    // Convert decoded f32 back to u8 for comparison
    // (decoded is already in sRGB, just scale to 0-255)
    let decoded_rgb: Vec<[u8; 3]> = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                let r = (decoded.get(x, y, 0) * 255.0).clamp(0.0, 255.0) as u8;
                let g = (decoded.get(x, y, 1) * 255.0).clamp(0.0, 255.0) as u8;
                let b = (decoded.get(x, y, 2) * 255.0).clamp(0.0, 255.0) as u8;
                [r, g, b]
            })
        })
        .collect();

    let src = ImgVec::new(original_rgb, width, height);
    let dst = ImgVec::new(decoded_rgb, width, height);

    // compute_ssimulacra2 handles sRGB->linear conversion internally
    compute_ssimulacra2(src.as_ref(), dst.as_ref()).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_encoding_mode() {
        // This test verifies the parser itself works correctly
        // We'll generate known bitstreams and verify parsing

        // For now, just ensure it doesn't panic on various inputs
        let _ = parse_encoding_mode(&[]);
        let _ = parse_encoding_mode(&[0xFF, 0x0A]);
        let _ = parse_encoding_mode(&[0; 100]);
    }
}

/// Return a test output directory, creating it if possible.
///
/// Prefers `/mnt/v/output/jxl-encoder-rs/{subdir}` (persistent, visible from Windows).
/// Falls back to `$TMPDIR/jxl-encoder-rs/{subdir}` when that path is unavailable
/// (CI, Docker, other machines).
pub fn test_output_dir(subdir: &str) -> std::path::PathBuf {
    let preferred = std::path::PathBuf::from(format!("/mnt/v/output/jxl-encoder-rs/{subdir}"));
    if std::fs::create_dir_all(&preferred).is_ok() {
        return preferred;
    }
    let fallback = std::env::temp_dir().join(format!("jxl-encoder-rs/{subdir}"));
    let _ = std::fs::create_dir_all(&fallback);
    fallback
}

/// Write test output to the best available directory. Never panics.
pub fn save_test_output(subdir: &str, filename: &str, data: &[u8]) {
    let dir = test_output_dir(subdir);
    let path = dir.join(filename);
    match std::fs::write(&path, data) {
        Ok(()) => eprintln!("Saved {} bytes to {}", data.len(), path.display()),
        Err(e) => eprintln!("Could not save to {} ({})", path.display(), e),
    }
}
