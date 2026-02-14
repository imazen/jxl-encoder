// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! BitWriter for encoding JPEG XL bitstreams.
//!
//! Writes bits in little-endian order, least-significant-bit first within
//! each byte. This is the inverse of the BitReader used in decoding.

use crate::error::{Error, Result};

/// Maximum bits that can be written in a single call.
/// Matches the decoder's MAX_BITS_PER_CALL for symmetry.
pub const MAX_BITS_PER_CALL: usize = 56;

/// Writes bits into a growable byte buffer.
///
/// Bits are written in little-endian order, with the least significant bit
/// of each value written first. This matches the JXL bitstream format.
///
/// # Example
///
/// ```
/// use jxl_encoder::bit_writer::BitWriter;
///
/// let mut writer = BitWriter::new();
/// writer.write(8, 0x12).unwrap();
/// writer.write(4, 0x3).unwrap();
/// writer.write(4, 0x4).unwrap();
/// writer.zero_pad_to_byte();
///
/// let bytes = writer.finish();
/// assert_eq!(bytes, vec![0x12, 0x43]);
/// ```
#[derive(Debug, Clone)]
pub struct BitWriter {
    /// Output buffer containing written bytes.
    storage: Vec<u8>,
    /// Total number of bits written.
    bits_written: usize,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    /// Creates a new empty BitWriter.
    pub fn new() -> Self {
        Self {
            storage: Vec::new(),
            bits_written: 0,
        }
    }

    /// Creates a new BitWriter with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity_bytes` - Initial capacity in bytes.
    pub fn with_capacity(capacity_bytes: usize) -> Self {
        Self {
            storage: Vec::with_capacity(capacity_bytes),
            bits_written: 0,
        }
    }

    /// Returns the total number of bits written.
    #[inline]
    pub fn bits_written(&self) -> usize {
        self.bits_written
    }

    /// Returns the number of bytes written (rounded up).
    #[inline]
    pub fn bytes_written(&self) -> usize {
        self.bits_written.div_ceil(8)
    }

    /// Returns true if the writer is aligned to a byte boundary.
    #[inline]
    pub fn is_byte_aligned(&self) -> bool {
        self.bits_written.is_multiple_of(8)
    }

    /// Returns the number of bits needed to reach the next byte boundary.
    #[inline]
    pub fn bits_to_byte_boundary(&self) -> usize {
        if self.bits_written.is_multiple_of(8) {
            0
        } else {
            8 - (self.bits_written % 8)
        }
    }

    /// Ensures the storage has capacity for at least `additional_bits` more bits.
    fn ensure_capacity(&mut self, additional_bits: usize) -> Result<()> {
        let total_bits = self.bits_written + additional_bits;
        let required_bytes = total_bits.div_ceil(8) + 8; // Extra 8 bytes for unaligned writes

        if self.storage.len() < required_bytes {
            self.storage
                .try_reserve(required_bytes - self.storage.len())?;
            self.storage.resize(required_bytes, 0);
        }
        Ok(())
    }

    /// Writes up to 56 bits to the buffer.
    ///
    /// Bits are written in little-endian order, least-significant-bit first.
    /// The value must fit in `n_bits` bits.
    ///
    /// # Arguments
    ///
    /// * `n_bits` - Number of bits to write (0-56).
    /// * `bits` - The value to write. Only the lower `n_bits` are used.
    ///
    /// # Errors
    ///
    /// Returns an error if `n_bits > 56` or if allocation fails.
    #[inline]
    pub fn write(&mut self, n_bits: usize, bits: u64) -> Result<()> {
        if n_bits > MAX_BITS_PER_CALL {
            return Err(Error::TooManyBitsPerCall(n_bits));
        }

        if n_bits == 0 {
            return Ok(());
        }

        debug_assert!(
            bits >> n_bits == 0 || n_bits == 64,
            "bits {bits:#x} has more than {n_bits} bits"
        );

        self.ensure_capacity(n_bits)?;

        let byte_offset = self.bits_written / 8;
        let bits_in_first_byte = self.bits_written % 8;

        // Shift the bits to align with the current position
        let shifted_bits = bits << bits_in_first_byte;

        // Read current value, OR in new bits, write back
        // This handles the case where we're continuing a partial byte
        let p = &mut self.storage[byte_offset..];

        // Use little-endian 64-bit write for efficiency
        // This may write more bytes than strictly necessary, but the extra
        // bytes are already zero-initialized
        let mut current = u64::from_le_bytes(p[..8].try_into().unwrap());
        current |= shifted_bits;
        p[..8].copy_from_slice(&current.to_le_bytes());

        self.bits_written += n_bits;
        Ok(())
    }

    /// Writes zeros to pad to the next byte boundary.
    ///
    /// If already byte-aligned, this is a no-op.
    pub fn zero_pad_to_byte(&mut self) {
        let remainder = self.bits_to_byte_boundary();
        if remainder > 0 {
            // We know this won't fail since remainder <= 7
            let _ = self.write(remainder, 0);
        }
        debug_assert!(self.is_byte_aligned());
    }

    /// Appends byte-aligned data from a slice.
    ///
    /// The writer must be byte-aligned before calling this method.
    ///
    /// # Errors
    ///
    /// Returns an error if the writer is not byte-aligned.
    pub fn append_bytes(&mut self, data: &[u8]) -> Result<()> {
        if !self.is_byte_aligned() {
            return Err(Error::NotByteAligned(self.bits_written));
        }

        if data.is_empty() {
            return Ok(());
        }

        let byte_offset = self.bits_written / 8;
        let new_len = byte_offset + data.len() + 8; // Extra padding for future writes

        if self.storage.len() < new_len {
            self.storage.try_reserve(new_len - self.storage.len())?;
            self.storage.resize(new_len, 0);
        }

        self.storage[byte_offset..byte_offset + data.len()].copy_from_slice(data);
        self.bits_written += data.len() * 8;

        // Ensure trailing zero for next write
        if byte_offset + data.len() < self.storage.len() {
            self.storage[byte_offset + data.len()] = 0;
        }

        Ok(())
    }

    /// Appends another BitWriter's contents.
    ///
    /// Both writers must be byte-aligned.
    ///
    /// # Errors
    ///
    /// Returns an error if either writer is not byte-aligned.
    pub fn append_byte_aligned(&mut self, other: &BitWriter) -> Result<()> {
        if !self.is_byte_aligned() {
            return Err(Error::NotByteAligned(self.bits_written));
        }
        if !other.is_byte_aligned() {
            return Err(Error::NotByteAligned(other.bits_written));
        }

        let other_bytes = other.bytes_written();
        self.append_bytes(&other.storage[..other_bytes])
    }

    /// Appends another BitWriter's contents, allowing unaligned data.
    ///
    /// This is slower than `append_byte_aligned` but works with any alignment.
    pub fn append_unaligned(&mut self, other: &BitWriter) -> Result<()> {
        let full_bytes = other.bits_written / 8;
        let remaining_bits = other.bits_written % 8;

        for &byte in &other.storage[..full_bytes] {
            self.write(8, byte as u64)?;
        }

        if remaining_bits > 0 {
            let mask = (1u64 << remaining_bits) - 1;
            let last_bits = other.storage[full_bytes] as u64 & mask;
            self.write(remaining_bits, last_bits)?;
        }

        Ok(())
    }

    /// Returns a view of the written bytes.
    ///
    /// The writer must be byte-aligned.
    ///
    /// # Panics
    ///
    /// Panics if the writer is not byte-aligned.
    pub fn as_bytes(&self) -> &[u8] {
        assert!(
            self.is_byte_aligned(),
            "BitWriter must be byte-aligned to get bytes"
        );
        &self.storage[..self.bytes_written()]
    }

    /// Returns a view of the internal storage for debugging.
    ///
    /// Unlike `as_bytes()`, this does not require byte alignment and returns
    /// the raw storage including any partial bytes. Useful for debugging
    /// bit-level encoding issues.
    pub fn peek_bytes(&self) -> &[u8] {
        let bytes = self.bits_written.div_ceil(8);
        &self.storage[..bytes.min(self.storage.len())]
    }

    /// Consumes the writer and returns the written bytes.
    ///
    /// The writer must be byte-aligned.
    ///
    /// # Panics
    ///
    /// Panics if the writer is not byte-aligned.
    pub fn finish(mut self) -> Vec<u8> {
        assert!(
            self.is_byte_aligned(),
            "BitWriter must be byte-aligned to finish"
        );
        self.storage.truncate(self.bytes_written());
        self.storage
    }

    /// Consumes the writer and returns the written bytes, padding if necessary.
    ///
    /// Unlike `finish`, this will zero-pad to byte alignment if needed.
    pub fn finish_with_padding(mut self) -> Vec<u8> {
        self.zero_pad_to_byte();
        self.storage.truncate(self.bytes_written());
        self.storage
    }
}

// Convenience write methods for common types
impl BitWriter {
    /// Writes a single bit (0 or 1).
    #[inline]
    pub fn write_bit(&mut self, bit: bool) -> Result<()> {
        self.write(1, bit as u64)
    }

    /// Writes an 8-bit unsigned integer.
    #[inline]
    pub fn write_u8(&mut self, value: u8) -> Result<()> {
        self.write(8, value as u64)
    }

    /// Writes a 16-bit unsigned integer in little-endian order.
    #[inline]
    pub fn write_u16(&mut self, value: u16) -> Result<()> {
        self.write(16, value as u64)
    }

    /// Writes a 32-bit unsigned integer in little-endian order.
    #[inline]
    pub fn write_u32(&mut self, value: u32) -> Result<()> {
        self.write(32, value as u64)
    }

    /// Writes a U32 value using the JXL variable-length encoding.
    ///
    /// The encoding is selector-based:
    /// - 0: value is `d0`
    /// - 1: value is `d1`
    /// - 2: value is `d2`
    /// - 3: `u_bits` bits follow, value is `d3 + read_bits`
    ///
    /// # Arguments
    ///
    /// * `value` - The value to encode.
    /// * `d0`, `d1`, `d2`, `d3` - Direct values for selectors 0-2 and offset for selector 3.
    /// * `u_bits` - Number of bits for the variable portion (selector 3).
    pub fn write_u32_coder(
        &mut self,
        value: u32,
        d0: u32,
        d1: u32,
        d2: u32,
        d3: u32,
        u_bits: usize,
    ) -> Result<()> {
        if value == d0 {
            self.write(2, 0)?;
        } else if value == d1 {
            self.write(2, 1)?;
        } else if value == d2 {
            self.write(2, 2)?;
        } else {
            debug_assert!(value >= d3, "value {value} < d3 {d3}");
            debug_assert!(
                (value - d3) < (1 << u_bits),
                "value {value} - d3 {d3} doesn't fit in {u_bits} bits"
            );
            self.write(2, 3)?;
            self.write(u_bits, (value - d3) as u64)?;
        }
        Ok(())
    }

    /// Writes an enum value using the jxl-rs default u2S encoding.
    /// This uses u2S(0, 1, Bits(4)+2, Bits(6)+18):
    /// - selector 0 → value 0
    /// - selector 1 → value 1
    /// - selector 2 → 2 + Bits(4) = values 2-17
    /// - selector 3 → 18 + Bits(6) = values 18-81
    pub fn write_enum_default(&mut self, value: u32) -> Result<()> {
        if value == 0 {
            self.write(2, 0)?;
        } else if value == 1 {
            self.write(2, 1)?;
        } else if value < 18 {
            self.write(2, 2)?;
            self.write(4, (value - 2) as u64)?;
        } else {
            debug_assert!(
                value < 82,
                "value {value} too large for default enum encoding"
            );
            self.write(2, 3)?;
            self.write(6, (value - 18) as u64)?;
        }
        Ok(())
    }

    /// Writes a U64 value using the JXL variable-length encoding.
    ///
    /// The encoding depends on the selector:
    /// - 0: value is 0
    /// - 1: 4 bits follow, value is 1 + read_bits (1-16)
    /// - 2: 8 bits follow, value is 17 + read_bits (17-272)
    /// - 3: 12 bits follow, then:
    ///   - If high bit is 0: value is 273 + low 12 bits (273-4368)
    ///   - If high bit is 1: 32 more bits follow, value is combined
    pub fn write_u64_coder(&mut self, value: u64) -> Result<()> {
        if value == 0 {
            self.write(2, 0)?;
        } else if value <= 16 {
            self.write(2, 1)?;
            self.write(4, value - 1)?;
        } else if value <= 272 {
            self.write(2, 2)?;
            self.write(8, value - 17)?;
        } else if value <= 4368 {
            self.write(2, 3)?;
            self.write(12, value - 273)?;
        } else {
            // Large value: selector 3 with shift indicator
            self.write(2, 3)?;
            let low = (value - 273) & 0xFFF;
            let high = (value - 273) >> 12;
            self.write(12, low | 0x1000)?; // Set high bit to indicate more data
            self.write(32, high)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_simple() {
        let mut writer = BitWriter::new();
        writer.write(8, 0x12).unwrap();
        writer.write(8, 0x34).unwrap();

        let bytes = writer.finish();
        assert_eq!(bytes, vec![0x12, 0x34]);
    }

    #[test]
    fn test_write_partial_bytes() {
        let mut writer = BitWriter::new();
        writer.write(4, 0x2).unwrap(); // Lower nibble
        writer.write(4, 0x1).unwrap(); // Upper nibble
        // Result: 0x12 (little-endian, LSB first)

        let bytes = writer.finish();
        assert_eq!(bytes, vec![0x12]);
    }

    #[test]
    fn test_write_across_bytes() {
        let mut writer = BitWriter::new();
        writer.write(4, 0x2).unwrap();
        writer.write(8, 0x34).unwrap();
        writer.write(4, 0x1).unwrap();

        let bytes = writer.finish();
        // Bits: 0010 | 0011_0100 | 0001
        // Byte 0: 0010 + lower 4 of 0x34 (0100) = 0100_0010 = 0x42
        // Byte 1: upper 4 of 0x34 (0011) + 0001 = 0001_0011 = 0x13
        assert_eq!(bytes, vec![0x42, 0x13]);
    }

    #[test]
    fn test_zero_pad() {
        let mut writer = BitWriter::new();
        writer.write(5, 0x15).unwrap();
        assert!(!writer.is_byte_aligned());
        assert_eq!(writer.bits_to_byte_boundary(), 3);

        writer.zero_pad_to_byte();
        assert!(writer.is_byte_aligned());

        let bytes = writer.finish();
        assert_eq!(bytes, vec![0x15]); // Lower 5 bits of 0x15, padded with zeros
    }

    #[test]
    fn test_append_bytes() {
        let mut writer = BitWriter::new();
        writer.write(8, 0x12).unwrap();
        writer.append_bytes(&[0x34, 0x56]).unwrap();

        let bytes = writer.finish();
        assert_eq!(bytes, vec![0x12, 0x34, 0x56]);
    }

    #[test]
    fn test_append_bytes_unaligned_fails() {
        let mut writer = BitWriter::new();
        writer.write(4, 0x2).unwrap();

        let result = writer.append_bytes(&[0x34]);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_too_many_bits() {
        let mut writer = BitWriter::new();
        let result = writer.write(57, 0);
        assert!(matches!(result, Err(Error::TooManyBitsPerCall(57))));
    }

    #[test]
    fn test_bits_written() {
        let mut writer = BitWriter::new();
        assert_eq!(writer.bits_written(), 0);

        writer.write(5, 0).unwrap();
        assert_eq!(writer.bits_written(), 5);

        writer.write(11, 0).unwrap();
        assert_eq!(writer.bits_written(), 16);
    }

    #[test]
    fn test_append_byte_aligned() {
        let mut writer1 = BitWriter::new();
        writer1.write(8, 0x12).unwrap();

        let mut writer2 = BitWriter::new();
        writer2.write(16, 0x5634).unwrap();

        writer1.append_byte_aligned(&writer2).unwrap();

        let bytes = writer1.finish();
        assert_eq!(bytes, vec![0x12, 0x34, 0x56]);
    }

    #[test]
    fn test_append_unaligned() {
        let mut writer1 = BitWriter::new();
        writer1.write(4, 0x2).unwrap();

        let mut writer2 = BitWriter::new();
        writer2.write(8, 0x34).unwrap();

        writer1.append_unaligned(&writer2).unwrap();
        writer1.zero_pad_to_byte();

        let bytes = writer1.finish();
        // 4 bits: 0010
        // 8 bits: 0011_0100
        // Result: 0010 + 0100 = 0100_0010 = 0x42, then 0011 padded = 0x03
        assert_eq!(bytes, vec![0x42, 0x03]);
    }

    #[test]
    fn test_finish_with_padding() {
        let mut writer = BitWriter::new();
        writer.write(5, 0x15).unwrap();

        let bytes = writer.finish_with_padding();
        assert_eq!(bytes, vec![0x15]);
    }

    #[test]
    fn test_u32_coder() {
        // Test direct values
        let mut writer = BitWriter::new();
        writer.write_u32_coder(0, 0, 1, 2, 3, 8).unwrap();
        writer.zero_pad_to_byte();
        assert_eq!(writer.as_bytes(), &[0b00]); // selector 0

        let mut writer = BitWriter::new();
        writer.write_u32_coder(1, 0, 1, 2, 3, 8).unwrap();
        writer.zero_pad_to_byte();
        assert_eq!(writer.as_bytes(), &[0b01]); // selector 1

        let mut writer = BitWriter::new();
        writer.write_u32_coder(2, 0, 1, 2, 3, 8).unwrap();
        writer.zero_pad_to_byte();
        assert_eq!(writer.as_bytes(), &[0b10]); // selector 2

        // Test variable encoding
        let mut writer = BitWriter::new();
        writer.write_u32_coder(10, 0, 1, 2, 3, 8).unwrap(); // 10 - 3 = 7
        writer.zero_pad_to_byte();
        // selector 3 (0b11) + 7 (0b0000_0111) in 8 bits
        // LSB first: bits are 11, then 11100000 (7 in 8 bits, LSB first)
        // Byte 0: 11 + 000001 (6 bits of 7) = 00011111 = 0x1F
        // Byte 1: remaining 00 = 0x00
        // Actually: selector 11, then value 7 = 00000111
        // Combined: 11 + 00000111 = 0b0000011111 -> bytes [0x1F, 0x00]
        assert_eq!(writer.as_bytes(), &[0x1F, 0x00]);
    }
}
