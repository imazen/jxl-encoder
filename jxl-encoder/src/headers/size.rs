// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Size encoding utilities for JPEG XL.

use crate::bit_writer::BitWriter;
use crate::error::Result;

/// Writes a size value using the JXL SizeHeader encoding.
///
/// The encoding is:
/// - Selector 0: 9 bits (values 0-511)
/// - Selector 1: 13 bits + 9 (values 9-8200)
/// - Selector 2: 18 bits + 8201 (values 8201-270536)
/// - Selector 3: 30 bits + 270537 (large values)
pub fn write_size(writer: &mut BitWriter, value: u32) -> Result<()> {
    if value < (1 << 9) {
        writer.write(2, 0)?;
        writer.write(9, value as u64)?;
    } else if value < (1 << 13) + (1 << 9) {
        writer.write(2, 1)?;
        writer.write(13, (value - (1 << 9)) as u64)?;
    } else if value < (1 << 18) + (1 << 13) + (1 << 9) {
        writer.write(2, 2)?;
        writer.write(18, (value - (1 << 13) - (1 << 9)) as u64)?;
    } else {
        writer.write(2, 3)?;
        writer.write(30, (value - (1 << 18) - (1 << 13) - (1 << 9)) as u64)?;
    }
    Ok(())
}

/// Computes the number of bits needed to encode a size value.
pub fn size_bits(value: u32) -> usize {
    if value < (1 << 9) {
        2 + 9
    } else if value < (1 << 13) + (1 << 9) {
        2 + 13
    } else if value < (1 << 18) + (1 << 13) + (1 << 9) {
        2 + 18
    } else {
        2 + 30
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_sizes() {
        for value in [0, 1, 100, 511] {
            let mut writer = BitWriter::new();
            write_size(&mut writer, value).unwrap();
            assert_eq!(writer.bits_written(), 11); // 2 + 9 bits
            assert_eq!(size_bits(value), 11);
        }
    }

    #[test]
    fn test_medium_sizes() {
        // Selector 1: 13 bits + 512 offset (values 512-8703)
        let value = 1000;
        let mut writer = BitWriter::new();
        write_size(&mut writer, value).unwrap();
        assert_eq!(writer.bits_written(), 15); // 2 + 13 bits
        assert_eq!(size_bits(value), 15);
    }

    #[test]
    fn test_large_sizes() {
        // Selector 2: 18 bits + 8704 offset (values 8704-270847)
        let value = 10000;
        let mut writer = BitWriter::new();
        write_size(&mut writer, value).unwrap();
        assert_eq!(writer.bits_written(), 20); // 2 + 18 bits
        assert_eq!(size_bits(value), 20);
    }

    #[test]
    fn test_very_large_sizes() {
        // Selector 3: 30 bits + 270848 offset
        let value = 300000;
        let mut writer = BitWriter::new();
        write_size(&mut writer, value).unwrap();
        assert_eq!(writer.bits_written(), 32); // 2 + 30 bits
        assert_eq!(size_bits(value), 32);
    }

    #[test]
    fn test_size_boundaries() {
        // Test boundary values for each selector
        let boundaries = [
            (0, 11),                        // min selector 0
            (511, 11),                      // max selector 0
            (512, 15),                      // min selector 1
            ((1 << 13) + (1 << 9) - 1, 15), // max selector 1
            ((1 << 13) + (1 << 9), 20),     // min selector 2
        ];

        for (value, expected_bits) in boundaries {
            let mut writer = BitWriter::new();
            write_size(&mut writer, value).unwrap();
            assert_eq!(
                writer.bits_written(),
                expected_bits,
                "value {} should use {} bits",
                value,
                expected_bits
            );
            assert_eq!(size_bits(value), expected_bits);
        }
    }
}
