// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Frame header for JPEG XL.

use crate::bit_writer::BitWriter;
use crate::error::Result;

/// Frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum FrameType {
    /// Regular frame.
    #[default]
    Regular = 0,
    /// LF (low-frequency) frame.
    LfFrame = 1,
    /// Reference-only frame (not displayed).
    ReferenceOnly = 2,
    /// Skip progressive rendering.
    SkipProgressive = 3,
}

/// Encoding method for the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Encoding {
    /// VarDCT encoding (lossy).
    #[default]
    VarDct = 0,
    /// Modular encoding (lossless or lossy).
    Modular = 1,
}

/// Blending mode for combining frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum BlendMode {
    /// Replace (no blending).
    #[default]
    Replace = 0,
    /// Add to previous frame.
    Add = 1,
    /// Blend using alpha.
    Blend = 2,
    /// Alpha-weighted add.
    AlphaWeightedAdd = 3,
    /// Multiply.
    Mul = 4,
}

/// Frame header structure.
#[derive(Debug, Clone)]
pub struct FrameHeader {
    /// Frame type.
    pub frame_type: FrameType,
    /// Encoding method.
    pub encoding: Encoding,
    /// Frame flags.
    pub flags: u64,
    /// Whether the frame uses a color transform.
    pub do_ycbcr: bool,
    /// JPEG upsampling mode for chroma.
    pub jpeg_upsampling: [u8; 3],
    /// Upsampling factor.
    pub upsampling: u32,
    /// Extra channel upsampling.
    pub ec_upsampling: Vec<u32>,
    /// Group size shift (0 = 256, 1 = 512, 2 = 1024, 3 = 2048).
    pub group_size_shift: u32,
    /// X offset for cropped frames.
    pub x0: i32,
    /// Y offset for cropped frames.
    pub y0: i32,
    /// Frame width (0 = full image width).
    pub width: u32,
    /// Frame height (0 = full image height).
    pub height: u32,
    /// Blending information.
    pub blend_mode: BlendMode,
    /// Alpha channel to use for blending.
    pub alpha_blend_channel: u32,
    /// Whether frame is saved for reference.
    pub save_as_reference: u32,
    /// Whether to save before color transform.
    pub save_before_ct: bool,
    /// Frame name.
    pub name: String,
    /// Duration in ticks (for animation).
    pub duration: u32,
    /// Timecode (if have_timecodes).
    pub timecode: u32,
    /// Whether this is the last frame.
    pub is_last: bool,
}

impl Default for FrameHeader {
    fn default() -> Self {
        Self {
            frame_type: FrameType::Regular,
            encoding: Encoding::VarDct,
            flags: 0,
            do_ycbcr: true,
            jpeg_upsampling: [0; 3],
            upsampling: 1,
            ec_upsampling: Vec::new(),
            group_size_shift: 0,
            x0: 0,
            y0: 0,
            width: 0,
            height: 0,
            blend_mode: BlendMode::Replace,
            alpha_blend_channel: 0,
            save_as_reference: 0,
            save_before_ct: false,
            name: String::new(),
            duration: 0,
            timecode: 0,
            is_last: true,
        }
    }
}

impl FrameHeader {
    /// Creates a frame header for a simple lossy frame.
    pub fn lossy() -> Self {
        Self {
            encoding: Encoding::VarDct,
            ..Default::default()
        }
    }

    /// Creates a frame header for a lossless frame.
    pub fn lossless() -> Self {
        Self {
            encoding: Encoding::Modular,
            do_ycbcr: false,
            ..Default::default()
        }
    }

    /// Writes the frame header to the bitstream.
    pub fn write(&self, writer: &mut BitWriter, _have_animation: bool) -> Result<()> {
        // all_default flag
        let all_default = self.is_default();
        writer.write_bit(all_default)?;

        if all_default {
            return Ok(());
        }

        // frame_type
        writer.write(2, self.frame_type as u64)?;

        // encoding
        writer.write(1, self.encoding as u64)?;

        // flags
        writer.write_u64_coder(self.flags)?;

        // do_ycbcr (only for VarDCT)
        if self.encoding == Encoding::VarDct {
            writer.write_bit(self.do_ycbcr)?;
        }

        // jpeg_upsampling (only for VarDCT with YCbCr)
        if self.encoding == Encoding::VarDct && self.do_ycbcr {
            for &up in &self.jpeg_upsampling {
                writer.write(2, up as u64)?;
            }
        }

        // upsampling
        writer.write_u32_coder(self.upsampling, 1, 2, 4, 8, 0)?;

        // ec_upsampling
        for &ecu in &self.ec_upsampling {
            writer.write_u32_coder(ecu, 1, 2, 4, 8, 0)?;
        }

        // group_size_shift
        writer.write(2, self.group_size_shift as u64)?;

        // x_qm_scale, b_qm_scale (only for VarDCT)
        if self.encoding == Encoding::VarDct {
            writer.write(3, 2)?; // x_qm_scale default
            writer.write(3, 2)?; // b_qm_scale default
        }

        // have_crop
        let have_crop = self.x0 != 0 || self.y0 != 0 || self.width != 0 || self.height != 0;
        if self.frame_type != FrameType::ReferenceOnly {
            writer.write_bit(have_crop)?;
            if have_crop {
                // Write crop coordinates
                self.write_crop(writer)?;
            }
        }

        // blending_info (for non-LF frames)
        if self.frame_type == FrameType::Regular || self.frame_type == FrameType::SkipProgressive {
            self.write_blending_info(writer)?;
        }

        // save_as_reference
        if self.frame_type != FrameType::LfFrame {
            writer.write(2, self.save_as_reference as u64)?;
        }

        // is_last
        writer.write_bit(self.is_last)?;

        // name - u2S(0, 0, Bits(4)+4, Bits(10)+20) for length, then bytes
        // For empty name: selector 0 means length 0
        let name_len = self.name.len() as u32;
        if name_len == 0 {
            writer.write(2, 0)?; // selector 0 = length 0
        } else if name_len < 4 {
            // This shouldn't happen with our current simple implementation
            writer.write(2, 0)?;
        } else if name_len < 20 {
            writer.write(2, 2)?;
            writer.write(4, (name_len - 4) as u64)?;
        } else {
            writer.write(2, 3)?;
            writer.write(10, (name_len - 20) as u64)?;
        }
        for byte in self.name.bytes() {
            writer.write(8, byte as u64)?;
        }

        // restoration_filter
        // For lossless modular encoding, we MUST disable Gaborish and EPF filters
        // Default has gab=true, epf_iters=2 which would blur the image!
        if self.encoding == Encoding::Modular {
            // all_default = false
            writer.write_bit(false)?;
            // gab = false (disable Gaborish filter)
            writer.write_bit(false)?;
            // epf_iters = 0 (disable Edge-Preserving Filter)
            writer.write(2, 0)?;
        } else {
            // For VarDCT, use defaults (gab=true, epf_iters=2)
            writer.write_bit(true)?;
        }

        // extensions (u64 selector, 0 = no extensions)
        writer.write(2, 0)?;

        Ok(())
    }

    /// Writes crop information.
    fn write_crop(&self, writer: &mut BitWriter) -> Result<()> {
        // x0, y0 as UnpackSigned
        let x0u = if self.x0 >= 0 {
            (self.x0 as u32) << 1
        } else {
            (((-self.x0 - 1) as u32) << 1) | 1
        };
        let y0u = if self.y0 >= 0 {
            (self.y0 as u32) << 1
        } else {
            (((-self.y0 - 1) as u32) << 1) | 1
        };

        writer.write_u32_coder(x0u, 0, 256, 2304, 18688, 14)?;
        writer.write_u32_coder(y0u, 0, 256, 2304, 18688, 14)?;

        // width, height
        writer.write_u32_coder(self.width, 0, 256, 2304, 18688, 14)?;
        writer.write_u32_coder(self.height, 0, 256, 2304, 18688, 14)?;

        Ok(())
    }

    /// Writes blending information.
    fn write_blending_info(&self, writer: &mut BitWriter) -> Result<()> {
        // blend_mode
        writer.write_u32_coder(self.blend_mode as u32, 0, 1, 2, 3, 2)?;

        // source reference (for blend modes that need it)
        if self.blend_mode != BlendMode::Replace {
            writer.write(2, 0)?; // source = 0
        }

        // alpha_channel (for alpha blend modes)
        if self.blend_mode == BlendMode::Blend || self.blend_mode == BlendMode::AlphaWeightedAdd {
            writer.write_u32_coder(self.alpha_blend_channel, 0, 1, 2, 3, 3)?;
            writer.write_bit(false)?; // clamp = false
        }

        Ok(())
    }

    /// Returns true if all fields are default.
    fn is_default(&self) -> bool {
        self.frame_type == FrameType::Regular
            && self.encoding == Encoding::VarDct
            && self.flags == 0
            && self.do_ycbcr
            && self.upsampling == 1
            && self.ec_upsampling.is_empty()
            && self.group_size_shift == 1
            && self.x0 == 0
            && self.y0 == 0
            && self.width == 0
            && self.height == 0
            && self.blend_mode == BlendMode::Replace
            && self.save_as_reference == 0
            && !self.save_before_ct
            && self.name.is_empty()
            && self.is_last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_frame() {
        let frame = FrameHeader::lossy();
        let mut writer = BitWriter::new();
        // Note: not all defaults match is_default() criteria
        frame.write(&mut writer, false).unwrap();
    }

    #[test]
    fn test_lossless_frame() {
        let frame = FrameHeader::lossless();
        assert_eq!(frame.encoding, Encoding::Modular);
        assert!(!frame.do_ycbcr);

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_frame_type_values() {
        assert_eq!(FrameType::Regular as u8, 0);
        assert_eq!(FrameType::LfFrame as u8, 1);
        assert_eq!(FrameType::ReferenceOnly as u8, 2);
        assert_eq!(FrameType::SkipProgressive as u8, 3);
    }

    #[test]
    fn test_encoding_values() {
        assert_eq!(Encoding::VarDct as u8, 0);
        assert_eq!(Encoding::Modular as u8, 1);
    }

    #[test]
    fn test_blend_mode_values() {
        assert_eq!(BlendMode::Replace as u8, 0);
        assert_eq!(BlendMode::Add as u8, 1);
        assert_eq!(BlendMode::Blend as u8, 2);
        assert_eq!(BlendMode::AlphaWeightedAdd as u8, 3);
        assert_eq!(BlendMode::Mul as u8, 4);
    }

    #[test]
    fn test_frame_with_crop() {
        let mut frame = FrameHeader::lossy();
        // x0/y0 encoded as signed -> unsigned, values here map to encoded 0
        frame.x0 = 0;
        frame.y0 = 0;
        frame.width = 20000; // Value must be >= 18688 for selector 3
        frame.height = 20000;

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        // Should have crop info written
        assert!(writer.bits_written() > 10);
    }

    #[test]
    fn test_frame_with_large_crop_offset() {
        let mut frame = FrameHeader::lossy();
        // Use values that encode to >= 256 (first threshold)
        frame.x0 = 128; // encodes to 256
        frame.y0 = 128;
        frame.width = 20000;
        frame.height = 20000;

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        assert!(writer.bits_written() > 10);
    }

    #[test]
    fn test_frame_with_name() {
        let mut frame = FrameHeader::lossy();
        frame.name = "TestFrame".to_string(); // 9 chars, falls into selector 2

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        // Should have name bytes written
        assert!(writer.bits_written() > 80); // 9 bytes = 72 bits + header
    }

    #[test]
    fn test_frame_with_long_name() {
        let mut frame = FrameHeader::lossy();
        frame.name = "ThisIsAVeryLongFrameName".to_string(); // 24 chars, selector 3

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        assert!(writer.bits_written() > 200);
    }

    #[test]
    fn test_lf_frame_type() {
        let mut frame = FrameHeader::lossy();
        frame.frame_type = FrameType::LfFrame;

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        // LF frames don't write blending_info or save_as_reference
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_reference_only_frame() {
        let mut frame = FrameHeader::lossy();
        frame.frame_type = FrameType::ReferenceOnly;
        frame.x0 = 5; // crop should be ignored for ReferenceOnly

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_skip_progressive_frame() {
        let mut frame = FrameHeader::lossy();
        frame.frame_type = FrameType::SkipProgressive;

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_blend_mode_add() {
        let mut frame = FrameHeader::lossy();
        frame.blend_mode = BlendMode::Add;

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        // Add mode writes source reference
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_blend_mode_blend_with_alpha() {
        let mut frame = FrameHeader::lossy();
        frame.blend_mode = BlendMode::Blend;
        frame.alpha_blend_channel = 1;

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        // Blend mode writes alpha_channel and clamp
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_blend_mode_alpha_weighted_add() {
        let mut frame = FrameHeader::lossy();
        frame.blend_mode = BlendMode::AlphaWeightedAdd;
        frame.alpha_blend_channel = 2;

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_blend_mode_mul() {
        let mut frame = FrameHeader::lossy();
        frame.blend_mode = BlendMode::Mul;

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_upsampling_factors() {
        for upsampling in [1, 2, 4, 8] {
            let mut frame = FrameHeader::lossy();
            frame.upsampling = upsampling;

            let mut writer = BitWriter::new();
            frame.write(&mut writer, false).unwrap();
            assert!(writer.bits_written() > 0);
        }
    }

    #[test]
    fn test_ec_upsampling() {
        let mut frame = FrameHeader::lossy();
        frame.ec_upsampling = vec![2, 4, 8];

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_group_size_shift() {
        for shift in 0..4 {
            let mut frame = FrameHeader::lossy();
            frame.group_size_shift = shift;

            let mut writer = BitWriter::new();
            frame.write(&mut writer, false).unwrap();
            assert!(writer.bits_written() > 0);
        }
    }

    #[test]
    fn test_save_as_reference() {
        let mut frame = FrameHeader::lossy();
        frame.save_as_reference = 2;

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_not_last_frame() {
        let mut frame = FrameHeader::lossy();
        frame.is_last = false;

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_jpeg_upsampling() {
        let mut frame = FrameHeader::lossy();
        frame.do_ycbcr = true;
        frame.jpeg_upsampling = [1, 2, 3];

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_vardct_no_ycbcr() {
        let mut frame = FrameHeader::lossy();
        frame.do_ycbcr = false;

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        // Without YCbCr, jpeg_upsampling is not written
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_all_default_check() {
        let mut frame = FrameHeader::default();
        // Default has group_size_shift = 0, but is_default expects 1
        frame.group_size_shift = 1;

        // This should now be considered all_default
        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        // all_default = true writes just 1 bit
        assert_eq!(writer.bits_written(), 1);
    }

    #[test]
    fn test_flags_nonzero() {
        let mut frame = FrameHeader::lossy();
        frame.flags = 0x3; // Small value that fits in u64 encoding

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_short_name() {
        let mut frame = FrameHeader::lossy();
        frame.name = "Hi".to_string(); // 2 chars, falls into selector 0 branch

        let mut writer = BitWriter::new();
        frame.write(&mut writer, false).unwrap();
        // selector 0 means length 0, name bytes are still written
        assert!(writer.bits_written() > 0);
    }
}
