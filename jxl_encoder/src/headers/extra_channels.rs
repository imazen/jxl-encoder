// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Extra channel definitions for JPEG XL.

use crate::bit_writer::BitWriter;
use crate::error::Result;

use super::file_header::BitDepth;

/// Type of extra channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ExtraChannelType {
    /// Alpha (transparency) channel.
    #[default]
    Alpha = 0,
    /// Depth map.
    Depth = 1,
    /// Spot color.
    SpotColor = 2,
    /// Selection mask.
    SelectionMask = 3,
    /// Black channel (for CMYK).
    Black = 4,
    /// CFA (Color Filter Array) channel.
    Cfa = 5,
    /// Thermal channel.
    Thermal = 6,
    /// Reserved for future use.
    Reserved0 = 7,
    Reserved1 = 8,
    Reserved2 = 9,
    Reserved3 = 10,
    Reserved4 = 11,
    Reserved5 = 12,
    Reserved6 = 13,
    Reserved7 = 14,
    /// Optional extra channel.
    Optional = 15,
}

/// Information about an extra channel.
#[derive(Debug, Clone)]
pub struct ExtraChannelInfo {
    /// Type of extra channel.
    pub ec_type: ExtraChannelType,
    /// Bit depth of this channel.
    pub bit_depth: BitDepth,
    /// Dimension shift (log2 of downsampling factor).
    pub dim_shift: u32,
    /// Name of the channel (optional).
    pub name: String,
    /// Whether alpha is premultiplied.
    pub alpha_associated: bool,
    /// Spot color values (for SpotColor type).
    pub spot_color: [f32; 4],
    /// CFA index (for CFA type).
    pub cfa_channel: u32,
}

impl Default for ExtraChannelInfo {
    fn default() -> Self {
        Self {
            ec_type: ExtraChannelType::Alpha,
            bit_depth: BitDepth::default(),
            dim_shift: 0,
            name: String::new(),
            alpha_associated: false,
            spot_color: [0.0; 4],
            cfa_channel: 0,
        }
    }
}

impl ExtraChannelInfo {
    /// Creates an alpha channel with default settings.
    pub fn alpha() -> Self {
        Self {
            ec_type: ExtraChannelType::Alpha,
            ..Default::default()
        }
    }

    /// Creates a depth channel.
    pub fn depth() -> Self {
        Self {
            ec_type: ExtraChannelType::Depth,
            ..Default::default()
        }
    }

    /// Creates a spot color channel.
    pub fn spot_color(color: [f32; 4]) -> Self {
        Self {
            ec_type: ExtraChannelType::SpotColor,
            spot_color: color,
            ..Default::default()
        }
    }

    /// Writes the extra channel info to the bitstream.
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        // d_alpha flag (true if this is default alpha)
        let d_alpha = self.is_default_alpha();
        writer.write_bit(d_alpha)?;

        if d_alpha {
            return Ok(());
        }

        // type
        writer.write_u32_coder(self.ec_type as u32, 0, 1, 2, 3, 4)?;

        // bit_depth
        self.bit_depth.write(writer)?;

        // dim_shift
        writer.write_u32_coder(self.dim_shift, 0, 3, 4, 1, 3)?;

        // name_len
        let name_len = self.name.len() as u32;
        writer.write_u32_coder(name_len, 0, 0, 0, 0, 10)?;
        for byte in self.name.bytes() {
            writer.write_u8(byte)?;
        }

        // alpha_associated (only for alpha channels)
        if self.ec_type == ExtraChannelType::Alpha {
            writer.write_bit(self.alpha_associated)?;
        }

        // spot_color (only for spot color channels)
        if self.ec_type == ExtraChannelType::SpotColor {
            for &value in &self.spot_color {
                writer.write_u32(value.to_bits())?;
            }
        }

        // cfa_channel (only for CFA channels)
        if self.ec_type == ExtraChannelType::Cfa {
            writer.write_u32_coder(self.cfa_channel, 1, 0, 2, 3, 4)?;
        }

        Ok(())
    }

    /// Returns true if this is a default alpha channel.
    fn is_default_alpha(&self) -> bool {
        self.ec_type == ExtraChannelType::Alpha
            && self.bit_depth.bits_per_sample == 8
            && !self.bit_depth.float_sample
            && self.dim_shift == 0
            && self.name.is_empty()
            && !self.alpha_associated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_alpha() {
        let alpha = ExtraChannelInfo::alpha();
        assert!(alpha.is_default_alpha());
    }

    #[test]
    fn test_write_default_alpha() {
        let alpha = ExtraChannelInfo::alpha();
        let mut writer = BitWriter::new();
        alpha.write(&mut writer).unwrap();
        writer.zero_pad_to_byte();

        // Default alpha should write just d_alpha=true (1 bit)
        assert_eq!(writer.bits_written(), 8); // Padded
    }

    #[test]
    fn test_non_default_alpha() {
        let mut alpha = ExtraChannelInfo::alpha();
        alpha.alpha_associated = true; // Makes it non-default
        assert!(!alpha.is_default_alpha());

        let mut writer = BitWriter::new();
        alpha.write(&mut writer).unwrap();
        // Should write d_alpha=false, type, bit_depth, dim_shift, name_len, alpha_associated
        assert!(writer.bits_written() > 1);
    }

    #[test]
    fn test_alpha_with_name() {
        let mut alpha = ExtraChannelInfo::alpha();
        alpha.name = "MyAlpha".to_string();
        assert!(!alpha.is_default_alpha());

        let mut writer = BitWriter::new();
        alpha.write(&mut writer).unwrap();
        // Should include name bytes
        assert!(writer.bits_written() > 8);
    }

    #[test]
    fn test_depth_channel() {
        let depth = ExtraChannelInfo::depth();
        assert_eq!(depth.ec_type, ExtraChannelType::Depth);

        let mut writer = BitWriter::new();
        depth.write(&mut writer).unwrap();
        // Not default alpha, so writes more data
        assert!(writer.bits_written() > 1);
    }

    #[test]
    fn test_spot_color_channel() {
        let spot = ExtraChannelInfo::spot_color([1.0, 0.5, 0.25, 1.0]);
        assert_eq!(spot.ec_type, ExtraChannelType::SpotColor);
        assert_eq!(spot.spot_color, [1.0, 0.5, 0.25, 1.0]);

        let mut writer = BitWriter::new();
        spot.write(&mut writer).unwrap();
        // Should include spot color values (4 x 32 bits = 128 bits)
        assert!(writer.bits_written() >= 128);
    }

    #[test]
    fn test_cfa_channel() {
        let cfa = ExtraChannelInfo {
            ec_type: ExtraChannelType::Cfa,
            cfa_channel: 2,
            ..Default::default()
        };

        let mut writer = BitWriter::new();
        cfa.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 1);
    }

    #[test]
    fn test_extra_channel_types() {
        // Test that all channel types have expected values
        assert_eq!(ExtraChannelType::Alpha as u8, 0);
        assert_eq!(ExtraChannelType::Depth as u8, 1);
        assert_eq!(ExtraChannelType::SpotColor as u8, 2);
        assert_eq!(ExtraChannelType::SelectionMask as u8, 3);
        assert_eq!(ExtraChannelType::Black as u8, 4);
        assert_eq!(ExtraChannelType::Cfa as u8, 5);
        assert_eq!(ExtraChannelType::Thermal as u8, 6);
        assert_eq!(ExtraChannelType::Optional as u8, 15);
    }

    #[test]
    fn test_dim_shift() {
        let mut alpha = ExtraChannelInfo::alpha();
        alpha.dim_shift = 2; // Downsampled by 4x
        assert!(!alpha.is_default_alpha());

        let mut writer = BitWriter::new();
        alpha.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 1);
    }
}
