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
///
/// Used by both VarDCT and Modular encoding paths. Fields are parameterized
/// to cover both modes. Use `lossy()` or `lossless()` constructors for defaults.
#[derive(Debug, Clone)]
pub struct FrameHeader {
    /// Frame type.
    pub frame_type: FrameType,
    /// Encoding method.
    pub encoding: Encoding,
    /// Whether the image metadata has xyb_encoded=true.
    /// Controls whether do_ycbcr is written (only when false).
    pub xyb_encoded: bool,
    /// Frame flags (e.g., SKIP_ADAPTIVE_LF_SMOOTHING=0x80, ENABLE_NOISE=0x01).
    pub flags: u64,
    /// Whether the frame uses YCbCr color transform (only written when !xyb_encoded).
    pub do_ycbcr: bool,
    /// JPEG upsampling mode for chroma (only for VarDCT + YCbCr).
    pub jpeg_upsampling: [u8; 3],
    /// Upsampling factor (1, 2, 4, or 8).
    pub upsampling: u32,
    /// Extra channel upsampling factors.
    pub ec_upsampling: Vec<u32>,
    /// Group size shift (Modular only: 0=128, 1=256, 2=512, 3=1024).
    pub group_size_shift: u32,
    /// X channel quant matrix scale (VarDCT only, 3 bits, range 0-7).
    pub x_qm_scale: u32,
    /// B channel quant matrix scale (VarDCT only, 3 bits, range 0-7).
    pub b_qm_scale: u32,
    /// Number of passes (1-10).
    pub num_passes: u32,
    /// X offset for cropped frames.
    pub x0: i32,
    /// Y offset for cropped frames.
    pub y0: i32,
    /// Frame width (0 = full image width).
    pub width: u32,
    /// Frame height (0 = full image height).
    pub height: u32,
    /// Blending information for the main frame.
    pub blend_mode: BlendMode,
    /// Per-extra-channel blending modes.
    pub ec_blend_modes: Vec<BlendMode>,
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
    /// Enable gaborish (Gabor-like blur in decoder loop filter).
    pub gaborish: bool,
    /// Number of EPF (Edge-Preserving Filter) iterations (0-3).
    pub epf_iters: u32,
}

impl Default for FrameHeader {
    fn default() -> Self {
        Self {
            frame_type: FrameType::Regular,
            encoding: Encoding::VarDct,
            xyb_encoded: true,
            flags: 0,
            do_ycbcr: false,
            jpeg_upsampling: [0; 3],
            upsampling: 1,
            ec_upsampling: Vec::new(),
            group_size_shift: 1,
            x_qm_scale: 2,
            b_qm_scale: 2,
            num_passes: 1,
            x0: 0,
            y0: 0,
            width: 0,
            height: 0,
            blend_mode: BlendMode::Replace,
            ec_blend_modes: Vec::new(),
            alpha_blend_channel: 0,
            save_as_reference: 0,
            save_before_ct: false,
            name: String::new(),
            duration: 0,
            timecode: 0,
            is_last: true,
            gaborish: true,
            epf_iters: 2,
        }
    }
}

impl FrameHeader {
    /// Creates a frame header for a lossy VarDCT frame with default parameters.
    ///
    /// Defaults: xyb_encoded=true, flags=SKIP_ADAPTIVE_LF_SMOOTHING (0x80),
    /// gaborish=true, epf_iters=2.
    pub fn lossy() -> Self {
        Self {
            encoding: Encoding::VarDct,
            xyb_encoded: true,
            flags: 0x80, // SKIP_ADAPTIVE_LF_SMOOTHING
            gaborish: true,
            epf_iters: 2,
            ..Default::default()
        }
    }

    /// Creates a frame header for a lossless Modular frame.
    ///
    /// Defaults: xyb_encoded=false, do_ycbcr=false, flags=0,
    /// group_size_shift=1 (256), gaborish=false, epf_iters=0.
    pub fn lossless() -> Self {
        Self {
            encoding: Encoding::Modular,
            xyb_encoded: false,
            do_ycbcr: false,
            flags: 0,
            group_size_shift: 1,
            gaborish: false,
            epf_iters: 0,
            ..Default::default()
        }
    }

    /// Writes the frame header to the bitstream.
    ///
    /// Follows the JXL codestream specification (ISO 18181-1) Table A.2.
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        // all_default: true only when all fields match the decoder's default
        // VarDCT default frame: Regular, VarDCT, no flags, do_ycbcr=true,
        // upsampling=1, group_size_shift=1, x/b_qm_scale=2, 1 pass,
        // no crop, Replace blend, is_last=true, no name, gab+epf2
        let all_default = self.is_all_default();
        writer.write_bit(all_default)?;
        if all_default {
            return Ok(());
        }

        // frame_type
        writer.write(2, self.frame_type as u64)?;

        // encoding
        writer.write(1, self.encoding as u64)?;

        // flags (U64)
        writer.write_u64_coder(self.flags)?;

        // do_ycbcr: only present when xyb_encoded is false
        if !self.xyb_encoded {
            writer.write_bit(self.do_ycbcr)?;
        }

        // jpeg_upsampling: only for VarDCT with YCbCr (when do_ycbcr and !xyb_encoded)
        if self.encoding == Encoding::VarDct && self.do_ycbcr && !self.xyb_encoded {
            for &up in &self.jpeg_upsampling {
                writer.write(2, up as u64)?;
            }
        }

        // upsampling (U32: 1, 2, 4, 8)
        writer.write_u32_coder(self.upsampling, 1, 2, 4, 8, 0)?;

        // ec_upsampling per extra channel
        for &ecu in &self.ec_upsampling {
            writer.write_u32_coder(ecu, 1, 2, 4, 8, 0)?;
        }

        // group_size_shift: Modular only (VarDCT uses fixed 256x256 groups)
        if self.encoding == Encoding::Modular {
            writer.write(2, self.group_size_shift as u64)?;
        }

        // x_qm_scale, b_qm_scale: VarDCT + xyb_encoded only
        if self.encoding == Encoding::VarDct && self.xyb_encoded {
            writer.write(3, self.x_qm_scale as u64)?;
            writer.write(3, self.b_qm_scale as u64)?;
        }

        // num_passes (U32: 1, 2, 3, 4+u(3))
        writer.write_u32_coder(self.num_passes, 1, 2, 3, 4, 3)?;
        // TODO: if num_passes > 1, write pass-specific data

        // have_crop (only for non-LfFrame, non-ReferenceOnly)
        if self.frame_type != FrameType::ReferenceOnly {
            let have_crop = self.x0 != 0 || self.y0 != 0 || self.width != 0 || self.height != 0;
            writer.write_bit(have_crop)?;
            if have_crop {
                self.write_crop(writer)?;
            }
        }

        // blending_info (for Regular or SkipProgressive frames)
        let normal_frame =
            self.frame_type == FrameType::Regular || self.frame_type == FrameType::SkipProgressive;
        if normal_frame {
            self.write_blending_info(writer)?;
        }

        // ec_blending_info per extra channel
        for &mode in &self.ec_blend_modes {
            // mode: U32(0, 1, 2, 3+u(2))
            writer.write_u32_coder(mode as u32, 0, 1, 2, 3, 2)?;
            // For full-frame Replace, no additional fields
        }

        // is_last (for Regular or SkipProgressive)
        if normal_frame {
            writer.write_bit(self.is_last)?;
        }

        // save_as_reference (only when !is_last and not LfFrame)
        if !self.is_last && self.frame_type != FrameType::LfFrame {
            writer.write(2, self.save_as_reference as u64)?;
        }

        // name
        self.write_name(writer)?;

        // restoration_filter (loop filter)
        self.write_loop_filter(writer)?;

        // frame header extensions (U64, always 0 for now)
        writer.write_u64_coder(0)?;

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
        writer.write_u32_coder(self.width, 0, 256, 2304, 18688, 14)?;
        writer.write_u32_coder(self.height, 0, 256, 2304, 18688, 14)?;

        Ok(())
    }

    /// Writes blending information for the main frame.
    fn write_blending_info(&self, writer: &mut BitWriter) -> Result<()> {
        writer.write_u32_coder(self.blend_mode as u32, 0, 1, 2, 3, 2)?;

        if self.blend_mode != BlendMode::Replace {
            writer.write(2, 0)?; // source = 0
        }

        if self.blend_mode == BlendMode::Blend || self.blend_mode == BlendMode::AlphaWeightedAdd {
            writer.write_u32_coder(self.alpha_blend_channel, 0, 1, 2, 3, 3)?;
            writer.write_bit(false)?; // clamp = false
        }

        Ok(())
    }

    /// Writes the frame name.
    fn write_name(&self, writer: &mut BitWriter) -> Result<()> {
        let name_len = self.name.len() as u32;
        if name_len == 0 {
            writer.write(2, 0)?; // selector 0 = length 0
        } else if name_len < 4 {
            writer.write(2, 0)?; // selector 0 (length encoded as 0, but name bytes follow)
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
        Ok(())
    }

    /// Writes the loop filter (restoration_filter) section.
    fn write_loop_filter(&self, writer: &mut BitWriter) -> Result<()> {
        // all_default means gab=true, epf_iters=2 (decoder defaults)
        let lf_all_default = self.gaborish && self.epf_iters == 2;

        writer.write_bit(lf_all_default)?;
        if lf_all_default {
            return Ok(());
        }

        // gab
        writer.write_bit(self.gaborish)?;
        if self.gaborish {
            writer.write_bit(false)?; // gab_custom = false (use default weights)
        }

        // epf_iters
        writer.write(2, self.epf_iters as u64)?;

        // EPF custom parameters (only when epf_iters > 0)
        if self.epf_iters > 0 {
            writer.write_bit(false)?; // epf_sharp_custom = false
            writer.write_bit(false)?; // epf_weight_custom = false
            writer.write_bit(false)?; // epf_sigma_custom = false
        }

        // loop filter extensions (U64)
        writer.write_u64_coder(0)?;

        Ok(())
    }

    /// Returns true if all fields match the decoder's "all_default" frame header.
    ///
    /// The all_default frame header is: Regular VarDCT, no flags, do_ycbcr=true,
    /// upsampling=1, group_size_shift=1, x/b_qm_scale=2, 1 pass, no crop,
    /// Replace blend, is_last=true, no name, default loop filter (gab+epf2).
    fn is_all_default(&self) -> bool {
        self.frame_type == FrameType::Regular
            && self.encoding == Encoding::VarDct
            && self.xyb_encoded
            && self.flags == 0
            && self.do_ycbcr
            && self.upsampling == 1
            && self.ec_upsampling.is_empty()
            && self.ec_blend_modes.is_empty()
            && self.group_size_shift == 1
            && self.x_qm_scale == 2
            && self.b_qm_scale == 2
            && self.num_passes == 1
            && self.x0 == 0
            && self.y0 == 0
            && self.width == 0
            && self.height == 0
            && self.blend_mode == BlendMode::Replace
            && self.save_as_reference == 0
            && !self.save_before_ct
            && self.name.is_empty()
            && self.is_last
            && self.gaborish
            && self.epf_iters == 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_frame() {
        let frame = FrameHeader::lossy();
        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
    }

    #[test]
    fn test_lossless_frame() {
        let frame = FrameHeader::lossless();
        assert_eq!(frame.encoding, Encoding::Modular);
        assert!(!frame.do_ycbcr);
        assert!(!frame.gaborish);
        assert_eq!(frame.epf_iters, 0);

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
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
        frame.x0 = 0;
        frame.y0 = 0;
        frame.width = 20000;
        frame.height = 20000;

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 10);
    }

    #[test]
    fn test_frame_with_large_crop_offset() {
        let mut frame = FrameHeader::lossy();
        frame.x0 = 128;
        frame.y0 = 128;
        frame.width = 20000;
        frame.height = 20000;

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 10);
    }

    #[test]
    fn test_frame_with_name() {
        let mut frame = FrameHeader::lossy();
        frame.name = "TestFrame".to_string();

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 80);
    }

    #[test]
    fn test_frame_with_long_name() {
        let mut frame = FrameHeader::lossy();
        frame.name = "ThisIsAVeryLongFrameName".to_string();

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 200);
    }

    #[test]
    fn test_lf_frame_type() {
        let mut frame = FrameHeader::lossy();
        frame.frame_type = FrameType::LfFrame;

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_reference_only_frame() {
        let mut frame = FrameHeader::lossy();
        frame.frame_type = FrameType::ReferenceOnly;

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_skip_progressive_frame() {
        let mut frame = FrameHeader::lossy();
        frame.frame_type = FrameType::SkipProgressive;

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_blend_mode_add() {
        let mut frame = FrameHeader::lossy();
        frame.blend_mode = BlendMode::Add;

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_blend_mode_blend_with_alpha() {
        let mut frame = FrameHeader::lossy();
        frame.blend_mode = BlendMode::Blend;
        frame.alpha_blend_channel = 1;

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_blend_mode_alpha_weighted_add() {
        let mut frame = FrameHeader::lossy();
        frame.blend_mode = BlendMode::AlphaWeightedAdd;
        frame.alpha_blend_channel = 2;

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_blend_mode_mul() {
        let mut frame = FrameHeader::lossy();
        frame.blend_mode = BlendMode::Mul;

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_upsampling_factors() {
        for upsampling in [1, 2, 4, 8] {
            let mut frame = FrameHeader::lossy();
            frame.upsampling = upsampling;

            let mut writer = BitWriter::new();
            frame.write(&mut writer).unwrap();
            assert!(writer.bits_written() > 0);
        }
    }

    #[test]
    fn test_ec_upsampling() {
        let mut frame = FrameHeader::lossy();
        frame.ec_upsampling = vec![2, 4, 8];

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_group_size_shift() {
        for shift in 0..4 {
            let mut frame = FrameHeader::lossless();
            frame.group_size_shift = shift;

            let mut writer = BitWriter::new();
            frame.write(&mut writer).unwrap();
            assert!(writer.bits_written() > 0);
        }
    }

    #[test]
    fn test_save_as_reference() {
        let mut frame = FrameHeader::lossy();
        frame.save_as_reference = 2;
        frame.is_last = false; // save_as_reference only written when !is_last

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_not_last_frame() {
        let mut frame = FrameHeader::lossy();
        frame.is_last = false;

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_vardct_loop_filter_all_default() {
        // gab=true, epf=2 → all_default for loop filter
        let frame = FrameHeader::lossy();
        assert!(frame.gaborish && frame.epf_iters == 2);

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
    }

    #[test]
    fn test_vardct_no_gaborish() {
        let mut frame = FrameHeader::lossy();
        frame.gaborish = false;
        frame.epf_iters = 1;

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_vardct_no_epf() {
        let mut frame = FrameHeader::lossy();
        frame.gaborish = true;
        frame.epf_iters = 0;

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_vardct_with_noise() {
        let mut frame = FrameHeader::lossy();
        frame.flags = 0x80 | 0x01; // SKIP_LF_SMOOTHING + ENABLE_NOISE

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_vardct_custom_qm_scale() {
        let mut frame = FrameHeader::lossy();
        frame.x_qm_scale = 5;
        frame.b_qm_scale = 4;

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_vardct_with_extra_channels() {
        let mut frame = FrameHeader::lossy();
        frame.ec_upsampling = vec![1]; // one extra channel, no upsampling
        frame.ec_blend_modes = vec![BlendMode::Replace];

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_lossless_with_extra_channels() {
        let mut frame = FrameHeader::lossless();
        frame.ec_upsampling = vec![1]; // one extra channel, no upsampling
        frame.ec_blend_modes = vec![BlendMode::Replace];

        let mut writer = BitWriter::new();
        frame.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    /// Verify that our VarDCT frame header matches the old hand-written write_frame_header()
    /// bit for bit. Parameters: x_qm=3, b_qm=2, epf=1, noise=false, gab=true, 0 extra channels.
    #[test]
    fn test_vardct_bit_exact_vs_old() {
        // Old path equivalent:
        // flags = 128 (0x80), x_qm=3, b_qm=2, epf=1, gab=true, 0 extra channels
        let mut old_writer = BitWriter::new();
        // Manually replicate the old write_frame_header():
        old_writer.write(1, 0).unwrap(); // not all_default
        old_writer.write(2, 0).unwrap(); // RegularFrame
        old_writer.write(1, 0).unwrap(); // VarDCT
        old_writer.write(2, 2).unwrap(); // flags U64 selector 2
        old_writer.write(8, 128 - 17).unwrap(); // flags = 128
        old_writer.write(2, 0).unwrap(); // upsampling = 1
        old_writer.write(3, 3).unwrap(); // x_qm_scale
        old_writer.write(3, 2).unwrap(); // b_qm_scale
        old_writer.write(2, 0).unwrap(); // num_passes = 1
        old_writer.write(1, 0).unwrap(); // have_crop = false
        old_writer.write(2, 0).unwrap(); // blend = Replace
        old_writer.write(1, 1).unwrap(); // is_last
        old_writer.write(2, 0).unwrap(); // name = ""
        // Loop filter: not all_default (gab=true but epf=1, not 2)
        old_writer.write(1, 0).unwrap(); // lf not all_default
        old_writer.write(1, 1).unwrap(); // gab = true
        old_writer.write(1, 0).unwrap(); // gab_custom = false
        old_writer.write(2, 1).unwrap(); // epf_iters = 1
        old_writer.write(1, 0).unwrap(); // epf_sharp_custom = false
        old_writer.write(1, 0).unwrap(); // epf_weight_custom = false
        old_writer.write(1, 0).unwrap(); // epf_sigma_custom = false
        old_writer.write(2, 0).unwrap(); // lf_extensions = 0
        old_writer.write(2, 0).unwrap(); // frame_extensions = 0

        let mut new_writer = BitWriter::new();
        let mut frame = FrameHeader::lossy();
        frame.x_qm_scale = 3;
        frame.b_qm_scale = 2;
        frame.epf_iters = 1;
        frame.write(&mut new_writer).unwrap();

        // Compare bit counts (writers may not be byte-aligned)
        assert_eq!(
            old_writer.bits_written(),
            new_writer.bits_written(),
            "VarDCT frame header bit count should match"
        );
        // Pad and compare bytes
        old_writer.zero_pad_to_byte();
        new_writer.zero_pad_to_byte();
        assert_eq!(
            old_writer.finish(),
            new_writer.finish(),
            "VarDCT frame header should be bit-exact"
        );
    }

    /// Verify VarDCT with gab=true, epf=2 (loop filter all_default).
    #[test]
    fn test_vardct_lf_all_default_bit_exact() {
        let mut old_writer = BitWriter::new();
        old_writer.write(1, 0).unwrap(); // not all_default
        old_writer.write(2, 0).unwrap(); // RegularFrame
        old_writer.write(1, 0).unwrap(); // VarDCT
        old_writer.write(2, 2).unwrap(); // flags U64 selector 2
        old_writer.write(8, 128 - 17).unwrap(); // flags = 128
        old_writer.write(2, 0).unwrap(); // upsampling = 1
        old_writer.write(3, 3).unwrap(); // x_qm_scale
        old_writer.write(3, 2).unwrap(); // b_qm_scale
        old_writer.write(2, 0).unwrap(); // num_passes = 1
        old_writer.write(1, 0).unwrap(); // have_crop = false
        old_writer.write(2, 0).unwrap(); // blend = Replace
        old_writer.write(1, 1).unwrap(); // is_last
        old_writer.write(2, 0).unwrap(); // name = ""
        old_writer.write(1, 1).unwrap(); // lf all_default
        old_writer.write(2, 0).unwrap(); // frame_extensions = 0

        let mut new_writer = BitWriter::new();
        let mut frame = FrameHeader::lossy();
        frame.x_qm_scale = 3;
        frame.b_qm_scale = 2;
        frame.gaborish = true;
        frame.epf_iters = 2;
        frame.write(&mut new_writer).unwrap();

        assert_eq!(
            old_writer.bits_written(),
            new_writer.bits_written(),
            "VarDCT lf all_default bit count should match"
        );
        old_writer.zero_pad_to_byte();
        new_writer.zero_pad_to_byte();
        assert_eq!(
            old_writer.finish(),
            new_writer.finish(),
            "VarDCT with lf all_default should be bit-exact"
        );
    }
}
