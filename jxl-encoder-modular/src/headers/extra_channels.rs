// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Extra channel definitions for JPEG XL.

use crate::bit_writer::BitWriter;
use crate::error::Result;

use super::file_header::BitDepth;

/// Write a name-string length using libjxl's `VisitNameString`
/// distribution: `U32(Val(0), Bits(4), BitsOffset(5, 16),
/// BitsOffset(10, 48))`. Caller writes the name bytes separately.
///
/// Range: 0..=1071 bytes.
fn write_name_string_length(writer: &mut BitWriter, len: u32) -> Result<()> {
    if len == 0 {
        writer.write(2, 0)?;
    } else if len < 16 {
        // selector 1 → Bits(4) raw value
        writer.write(2, 1)?;
        writer.write(4, len as u64)?;
    } else if len < 48 {
        // selector 2 → BitsOffset(5, 16)
        writer.write(2, 2)?;
        writer.write(5, (len - 16) as u64)?;
    } else {
        // selector 3 → BitsOffset(10, 48), max 48+1023 = 1071
        debug_assert!(
            len <= 48 + 1023,
            "extra-channel name length {len} exceeds spec maximum 1071"
        );
        writer.write(2, 3)?;
        writer.write(10, (len - 48) as u64)?;
    }
    Ok(())
}

/// Write a CFA channel index using libjxl's
/// `U32(Val(1), Bits(2), BitsOffset(4, 3), BitsOffset(8, 19))`.
///
/// Range: 0..=274.
fn write_cfa_channel(writer: &mut BitWriter, value: u32) -> Result<()> {
    if value == 1 {
        writer.write(2, 0)?;
    } else if value < 4 {
        // selector 1 → Bits(2): values 0, 2, 3 (1 already taken by selector 0)
        writer.write(2, 1)?;
        writer.write(2, value as u64)?;
    } else if value < 19 {
        // selector 2 → BitsOffset(4, 3): values 3-18 (3 + 0..15)
        // Note: value=3 fits both selector 1 (Bits(2) can't reach 3
        // when value is in {0,2,3}? — Bits(2) range is 0-3, so value=3
        // IS encodable via selector 1. Encoder picks the cheaper
        // selector; selector 1 is cheaper. We pick selector 2 only for
        // values that selector 1 cannot reach: 4-18.
        writer.write(2, 2)?;
        writer.write(4, (value - 3) as u64)?;
    } else {
        // selector 3 → BitsOffset(8, 19), max 19 + 255 = 274
        debug_assert!(
            value <= 19 + 255,
            "cfa_channel value {value} exceeds spec maximum 274"
        );
        writer.write(2, 3)?;
        writer.write(8, (value - 19) as u64)?;
    }
    Ok(())
}

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

        // type — Enum() distribution per spec, NOT Val/Val/Val/BitsOffset.
        // libjxl/jxl-rs use the default UnconditionalCoder for enum fields:
        // U32(Val(0), Val(1), BitsOffset(4, 2), BitsOffset(6, 18)). Alpha
        // (0) and Depth (1) encoded correctly by accident under the old
        // Val/Val/Val/BitsOffset coder; SpotColor (2) through Thermal (6)
        // produced wrong bits. Fixes #8 Bug A.
        writer.write_enum_default(self.ec_type as u32)?;

        // bit_depth
        self.bit_depth.write(writer)?;

        // dim_shift — U32(Val(0), Val(3), Val(4), BitsOffset(3, 1)).
        // Matches `u2S(0, 3, 4, Bits(3) + 1)` in jxl-rs's
        // ExtraChannelInfo.dim_shift attribute.
        writer.write_u32_coder(self.dim_shift, 0, 3, 4, 1, 3)?;

        // name_len — `U32(Val(0), Bits(4), BitsOffset(5, 16),
        // BitsOffset(10, 48))` per libjxl `VisitNameString`
        // (frame_header.h:35). Old encoder used (0, 0, 0, 0, 10) which
        // routed every length through selector 3 + 10 bits of (len - 0)
        // — wrong bit pattern for any non-zero length. Fixes #8 Bug D.
        let name_len = self.name.len() as u32;
        write_name_string_length(writer, name_len)?;
        for byte in self.name.bytes() {
            writer.write_u8(byte)?;
        }

        // alpha_associated (only for alpha channels)
        if self.ec_type == ExtraChannelType::Alpha {
            writer.write_bit(self.alpha_associated)?;
        }

        // spot_color (only for spot color channels) — per spec the four
        // spot color samples are written as F16, NOT as F32 IEEE 754.
        // libjxl uses `visitor->F16()`; we have `crate::f16::write_f16`.
        // The old `writer.write_u32(value.to_bits())` produced 4×32 bits
        // of garbage. Fixes #8 Bug B.
        if self.ec_type == ExtraChannelType::SpotColor {
            for &value in &self.spot_color {
                crate::f16::write_f16(value, writer)?;
            }
        }

        // cfa_channel (only for CFA channels) — `U32(Val(1), Bits(2),
        // BitsOffset(4, 3), BitsOffset(8, 19))` per libjxl + jxl-rs.
        // Old encoder used (1, 0, 2, 3, 4) which sent everything except
        // 1, 0, and 2 through selector 3 + 4 bits with offset 3 — wrong
        // for the common Bayer/Quad-Bayer indices. Fixes #8 Bug C.
        if self.ec_type == ExtraChannelType::Cfa {
            write_cfa_channel(writer, self.cfa_channel)?;
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
        // Spot color values are 4 × F16 = 64 bits. Header overhead
        // (d_alpha + ec_type enum + bit_depth + dim_shift + name_len)
        // is small but non-zero. Total > 64 + a few bits.
        assert!(
            writer.bits_written() > 64,
            "spot color encoding too short ({}) — header missing?",
            writer.bits_written(),
        );
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

    // ========== #8 Bug A: enum coder for ec_type ==========

    /// Verify each non-Alpha extra channel type encodes through
    /// `write_enum_default` (Val(0), Val(1), BitsOffset(4,2),
    /// BitsOffset(6,18)) — the spec-correct path.
    ///
    /// Pre-fix encoding used `write_u32_coder(_, 0, 1, 2, 3, 4)` which
    /// happens to match for Alpha (0) and Depth (1) but breaks for
    /// SpotColor (2) onward.
    #[test]
    fn test_enum_default_coder_for_ec_type() {
        // SpotColor=2: write_enum_default selector 2 + 4 bits of (2-2=0).
        // Pre-fix: selector 2 (matches d2=2), 0 extra bits — different total.
        for &(ec, name) in &[
            (ExtraChannelType::SpotColor, "SpotColor"),
            (ExtraChannelType::SelectionMask, "SelMask"),
            (ExtraChannelType::Black, "Black"),
            (ExtraChannelType::Cfa, "CFA"),
            (ExtraChannelType::Thermal, "Thermal"),
        ] {
            let info = ExtraChannelInfo {
                ec_type: ec,
                ..Default::default()
            };
            let mut writer = BitWriter::new();
            info.write(&mut writer).unwrap();
            // Should at minimum: d_alpha(1) + selector(2) + extra bits + dim_shift + bit_depth + name_len.
            // We don't check exact bit count but ensure encoding succeeded.
            assert!(
                writer.bits_written() > 4,
                "{name}: expected >4 bits, got {}",
                writer.bits_written(),
            );
        }
    }

    // ========== #8 Bug B: SpotColor uses F16, not F32 ==========

    #[test]
    fn test_spot_color_uses_f16() {
        let spot = ExtraChannelInfo::spot_color([0.0, 1.0, -1.0, 0.5]);
        let mut writer = BitWriter::new();
        spot.write(&mut writer).unwrap();
        // Pre-fix: 4 × 32-bit f32 = 128 bits for spot color alone.
        // Post-fix: 4 × 16-bit f16 = 64 bits for spot color alone.
        // Header overhead before spot_color: ~30-50 bits depending on
        // bit_depth + name_len + alpha_associated. Total post-fix should
        // be well under 128 bits, pre-fix would have been well over 128.
        assert!(
            writer.bits_written() < 128,
            "spot color now uses F16 — total bits should be <128, got {}",
            writer.bits_written(),
        );
        // And it should fit comfortably in 64 bits (F16) plus header.
        assert!(
            writer.bits_written() > 64,
            "spot color encoding too short ({}); header missing?",
            writer.bits_written(),
        );
    }

    // ========== #8 Bug C: CFA channel U32 distribution ==========

    #[test]
    fn test_cfa_channel_distribution() {
        // value=1 → selector 0, no extra bits.
        // value=0 → selector 1 + Bits(2)=00.
        // value=5 → selector 2 + Bits(4)=0010 (5-3=2).
        // value=20 → selector 3 + Bits(8)=00000001 (20-19=1).
        for cfa in [0u32, 1, 2, 3, 4, 5, 10, 18, 19, 20, 100, 274] {
            let info = ExtraChannelInfo {
                ec_type: ExtraChannelType::Cfa,
                cfa_channel: cfa,
                ..Default::default()
            };
            let mut writer = BitWriter::new();
            // Should not panic for any valid CFA value.
            info.write(&mut writer).unwrap();
            assert!(writer.bits_written() > 4);
        }
    }

    #[test]
    fn test_cfa_channel_value_1_is_shortest() {
        // value=1 is the Val(0) selector — only 2 bits for the
        // CFA-channel field. Other values need more.
        let info1 = ExtraChannelInfo {
            ec_type: ExtraChannelType::Cfa,
            cfa_channel: 1,
            ..Default::default()
        };
        let info100 = ExtraChannelInfo {
            ec_type: ExtraChannelType::Cfa,
            cfa_channel: 100,
            ..Default::default()
        };
        let mut w1 = BitWriter::new();
        let mut w100 = BitWriter::new();
        info1.write(&mut w1).unwrap();
        info100.write(&mut w100).unwrap();
        assert!(
            w1.bits_written() < w100.bits_written(),
            "cfa=1 should be shorter than cfa=100; got {} vs {}",
            w1.bits_written(),
            w100.bits_written(),
        );
    }

    // ========== #8 Bug D: name length distribution ==========

    #[test]
    fn test_name_length_distribution() {
        // Test that various name lengths encode without panicking.
        for &len in &[0usize, 1, 4, 15, 16, 32, 47, 48, 100, 1071] {
            let info = ExtraChannelInfo {
                ec_type: ExtraChannelType::Depth,
                name: "x".repeat(len),
                ..Default::default()
            };
            let mut writer = BitWriter::new();
            info.write(&mut writer).unwrap();
            // Sanity: name bytes contribute at least 8*len bits.
            assert!(
                writer.bits_written() >= 8 * len,
                "len={len}: bits_written {} < 8*len {}",
                writer.bits_written(),
                8 * len,
            );
        }
    }

    #[test]
    fn test_name_empty_is_shortest() {
        // Empty name: selector 0 → 2 bits, no name bytes.
        // 5-byte name: selector 1 + 4 bits + 5*8 bits.
        let info0 = ExtraChannelInfo {
            ec_type: ExtraChannelType::Depth,
            name: String::new(),
            ..Default::default()
        };
        let info5 = ExtraChannelInfo {
            ec_type: ExtraChannelType::Depth,
            name: "Hello".to_string(),
            ..Default::default()
        };
        let mut w0 = BitWriter::new();
        let mut w5 = BitWriter::new();
        info0.write(&mut w0).unwrap();
        info5.write(&mut w5).unwrap();
        // 5-byte name adds ~44 bits over empty (4 bits length + 40 bits chars).
        let delta = w5.bits_written() as i64 - w0.bits_written() as i64;
        assert!(
            (40..=50).contains(&delta),
            "5-byte name should add ~44 bits; got delta {delta}",
        );
    }

    #[test]
    fn test_write_name_string_length_helper() {
        // Verify the helper writes the expected number of bits per range.
        // 0 → 2 bits. 1-15 → 6 bits. 16-47 → 7 bits. 48-1071 → 12 bits.
        for &(len, expect) in &[
            (0u32, 2usize),
            (1, 6),
            (15, 6),
            (16, 7),
            (47, 7),
            (48, 12),
            (1071, 12),
        ] {
            let mut w = BitWriter::new();
            write_name_string_length(&mut w, len).unwrap();
            assert_eq!(
                w.bits_written(),
                expect,
                "len={len} should write {expect} bits, got {}",
                w.bits_written(),
            );
        }
    }

    #[test]
    fn test_write_cfa_channel_helper() {
        // value=1 → 2 bits (selector 0). 0,2,3 → 4 bits (selector 1+Bits(2)).
        // 4-18 → 6 bits (selector 2+Bits(4)). 19-274 → 10 bits (selector 3+Bits(8)).
        for &(v, expect) in &[
            (1u32, 2usize),
            (0, 4),
            (2, 4),
            (3, 4),
            (4, 6),
            (18, 6),
            (19, 10),
            (274, 10),
        ] {
            let mut w = BitWriter::new();
            write_cfa_channel(&mut w, v).unwrap();
            assert_eq!(
                w.bits_written(),
                expect,
                "cfa={v} should write {expect} bits, got {}",
                w.bits_written(),
            );
        }
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
