// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
        }
    }

    #[test]
    fn test_medium_sizes() {
        let value = 1000;
        let mut writer = BitWriter::new();
        write_size(&mut writer, value).unwrap();
        assert_eq!(writer.bits_written(), 15); // 2 + 13 bits
    }
}
