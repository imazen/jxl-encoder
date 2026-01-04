//! Test helpers to prevent false positives and verify what tests actually do.

use crate::error::Result;

/// Encoding mode in JXL
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingMode {
    VarDct,  // encoding=0 (lossy)
    Modular, // encoding=1 (lossless)
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
    for start_bit in 38..70 {
        let all_default = read_bit(data, start_bit)?;
        if all_default == 0 {
            // all_default=0, so frame_type (2 bits) and encoding (1 bit) follow
            let encoding_bit = read_bit(data, start_bit + 3)?;
            return Some(match encoding_bit {
                0 => EncodingMode::VarDct,
                1 => EncodingMode::Modular,
                _ => unreachable!(),
            });
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
/// Encodes with Modular, verifies encoding mode, then decodes.
pub fn test_lossless_roundtrip(
    data: &[u8],
    width: usize,
    height: usize,
    test_name: &str,
) -> Result<()> {
    let encoded = crate::encoder::encode_rgb8(data, width, height)?;

    // VERIFY we actually used Modular encoding
    assert_encoding_mode(&encoded, EncodingMode::Modular, test_name);

    // Decode and verify
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&encoded))
        .map_err(|e| {
            crate::error::Error::InvalidInput(format!("jxl-oxide decode failed: {:?}", e))
        })?;

    assert_eq!(image.width(), width as u32);
    assert_eq!(image.height(), height as u32);

    // Actually render to ensure full decode
    image.render_frame(0).map_err(|e| {
        crate::error::Error::InvalidInput(format!("jxl-oxide render failed: {:?}", e))
    })?;

    Ok(())
}

/// Standard roundtrip test for lossy VarDCT encoding.
/// Encodes with VarDCT, verifies encoding mode, then decodes.
pub fn test_lossy_roundtrip(
    data: &[u8],
    width: usize,
    height: usize,
    distance: f32,
    test_name: &str,
) -> Result<()> {
    let encoded = crate::encoder::encode_lossy_rgb8(data, width, height, distance)?;

    // Save for debugging
    let debug_path = format!("/tmp/{}.jxl", test_name);
    std::fs::write(&debug_path, &encoded).ok();
    eprintln!("DEBUG: Saved {} bytes to {}", encoded.len(), debug_path);

    // VERIFY we actually used VarDCT encoding
    assert_encoding_mode(&encoded, EncodingMode::VarDct, test_name);

    // Decode with jxl-oxide
    eprintln!("DEBUG: Trying jxl-oxide decoder...");
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&encoded))
        .map_err(|e| {
            crate::error::Error::InvalidInput(format!("jxl-oxide decode failed: {:?}", e))
        })?;

    assert_eq!(image.width(), width as u32);
    assert_eq!(image.height(), height as u32);

    // Actually render to ensure full decode
    image.render_frame(0).map_err(|e| {
        crate::error::Error::InvalidInput(format!("jxl-oxide render failed: {:?}", e))
    })?;

    Ok(())
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
        let _ = parse_encoding_mode(&vec![0; 100]);
    }
}
