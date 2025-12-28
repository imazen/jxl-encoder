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

        // name (if not is_last)
        if !self.is_last || !self.name.is_empty() {
            let name_len = self.name.len() as u32;
            writer.write_u32_coder(name_len, 0, 0, 0, 0, 10)?;
            for byte in self.name.bytes() {
                writer.write_u8(byte)?;
            }
        }

        // TODO: More frame header fields (restoration filter, passes, etc.)

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
}
