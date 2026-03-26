// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Frame header for JPEG XL.

use crate::bit_writer::BitWriter;
use crate::error::Result;

/// Crop rectangle for a frame within the canvas.
///
/// When set on a frame, the frame contains only the specified rectangular region.
/// The decoder composites this region onto the persistent canvas using the frame's
/// blend mode. For `Replace` blending, only the crop rectangle is replaced; the
/// rest of the canvas is unchanged.
#[derive(Debug, Clone, Copy)]
pub struct FrameCrop {
    /// X offset of the crop region within the canvas.
    pub x0: i32,
    /// Y offset of the crop region within the canvas.
    pub y0: i32,
    /// Width of the crop region.
    pub width: u32,
    /// Height of the crop region.
    pub height: u32,
}

/// Overrides for frame header fields in animation encoding.
///
/// Used by `encode_animation()` to set per-frame duration, is_last, and animation flags
/// without exposing the full FrameHeader construction to callers.
#[derive(Debug, Clone, Default)]
pub struct FrameOptions {
    /// Whether the file header has animation enabled.
    pub have_animation: bool,
    /// Whether the file header has have_timecodes enabled.
    pub have_timecodes: bool,
    /// Duration in ticks for this frame (only used if have_animation=true).
    pub duration: u32,
    /// Whether this is the last frame in the file.
    pub is_last: bool,
    /// Optional crop rectangle for this frame (None = full frame).
    pub crop: Option<FrameCrop>,
}

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

/// Frame flag: enable noise synthesis.
pub const ENABLE_NOISE: u64 = 0x01;
/// Frame flag: enable patches (dictionary-based repeated patterns).
pub const PATCHES_FLAG: u64 = 0x02;
/// Frame flag: enable splines.
pub const SPLINES_FLAG: u64 = 0x10;
/// Frame flag: use a separate LF frame for DC coefficients.
pub const USE_LF_FRAME: u64 = 0x20;
/// Frame flag: skip adaptive LF smoothing.
pub const SKIP_ADAPTIVE_LF_SMOOTHING: u64 = 0x80;

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
    /// Frame flags (bitfield: ENABLE_NOISE=0x01, PATCHES_FLAG=0x02,
    /// SKIP_ADAPTIVE_LF_SMOOTHING=0x80).
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
    /// Number of passes (1-11).
    pub num_passes: u32,
    /// Per-pass shift values (num_passes - 1 elements). Last pass implicitly has shift=0.
    /// Each shift is 0-3 bits: coefficients are right-shifted before encoding,
    /// left-shifted by the decoder before accumulation.
    pub pass_shifts: Vec<u32>,
    /// Number of downsampling brackets (0-4).
    pub num_ds: u32,
    /// Downsample factors per bracket (1, 2, 4, or 8).
    pub ds_downsample: Vec<u32>,
    /// Last pass index per downsampling bracket.
    pub ds_last_pass: Vec<u32>,
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
    /// Source reference frame for blending (0-3).
    pub blend_source: u32,
    /// Alpha channel to use for blending.
    pub alpha_blend_channel: u32,
    /// Whether frame is saved for reference.
    pub save_as_reference: u32,
    /// Whether to save before color transform.
    pub save_before_ct: bool,
    /// Frame name.
    pub name: String,
    /// Whether the file header signals animation (have_animation=true).
    /// When true, duration/timecode fields are written for normal frames.
    pub have_animation: bool,
    /// Whether the file header signals have_timecodes.
    pub have_timecodes: bool,
    /// Duration in ticks (for animation).
    pub duration: u32,
    /// Timecode (if have_timecodes).
    pub timecode: u32,
    /// Whether this is the last frame.
    pub is_last: bool,
    /// LF level for LfFrame (frame_type=1). Written as u2S(1,2,3,4).
    /// Only meaningful when frame_type == LfFrame. Typically 1 for DC frames.
    pub lf_level: u32,
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
            pass_shifts: Vec::new(),
            num_ds: 0,
            ds_downsample: Vec::new(),
            ds_last_pass: Vec::new(),
            x0: 0,
            y0: 0,
            width: 0,
            height: 0,
            blend_mode: BlendMode::Replace,
            blend_source: 0,
            ec_blend_modes: Vec::new(),
            alpha_blend_channel: 0,
            save_as_reference: 0,
            save_before_ct: false,
            name: String::new(),
            have_animation: false,
            have_timecodes: false,
            duration: 0,
            timecode: 0,
            is_last: true,
            lf_level: 0,
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

    /// Creates a frame header for an LF (DC) frame.
    ///
    /// LfFrames contain DC coefficients at 1/8 resolution, encoded as modular.
    /// Uses xyb_encoded=true, group_size_shift=1 (256), no loop filter.
    /// The `width` and `height` are the DC frame dimensions (xsize_blocks × ysize_blocks).
    pub fn lf_frame(width: u32, height: u32, lf_level: u32) -> Self {
        Self {
            frame_type: FrameType::LfFrame,
            encoding: Encoding::Modular,
            xyb_encoded: true,
            flags: SKIP_ADAPTIVE_LF_SMOOTHING,
            gaborish: false,
            epf_iters: 0,
            is_last: false,
            save_before_ct: false,
            width,
            height,
            lf_level,
            group_size_shift: 1, // 256
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

        // upsampling, ec_upsampling: not present when USE_LF_FRAME flag is set
        // (jxl-rs frame_header.rs:288-300)
        if self.flags & USE_LF_FRAME == 0 {
            // upsampling (U32: 1, 2, 4, 8)
            writer.write_u32_coder(self.upsampling, 1, 2, 4, 8, 0)?;

            // ec_upsampling per extra channel
            for &ecu in &self.ec_upsampling {
                writer.write_u32_coder(ecu, 1, 2, 4, 8, 0)?;
            }
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

        // num_passes: not present for ReferenceOnly frames (libjxl spec)
        if self.frame_type != FrameType::ReferenceOnly {
            // num_passes (U32: 1, 2, 3, 4+u(3))
            writer.write_u32_coder(self.num_passes, 1, 2, 3, 4, 3)?;
            if self.num_passes != 1 {
                self.write_passes(writer)?;
            }
        }

        // lf_level: only for LfFrame, written as u2S(1, 2, 3, 4)
        // (jxl-rs frame_header.rs:321-324, after passes, before have_crop)
        if self.frame_type == FrameType::LfFrame {
            writer.write_u32_coder(self.lf_level, 1, 2, 3, 4, 0)?;
        }

        // have_crop: present for all frame types except LfFrame
        if self.frame_type != FrameType::LfFrame {
            let have_crop = self.x0 != 0 || self.y0 != 0 || self.width != 0 || self.height != 0;
            writer.write_bit(have_crop)?;
            if have_crop {
                // x0, y0: only for Regular/SkipProgressive frames (not ReferenceOnly)
                if self.frame_type != FrameType::ReferenceOnly {
                    self.write_crop_origin(writer)?;
                }
                // width, height: always present when have_crop
                Self::write_crop_u32(writer, self.width)?;
                Self::write_crop_u32(writer, self.height)?;
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
            self.write_ec_blending_info(mode, writer)?;
        }

        // duration and timecode (for animated normal frames)
        if normal_frame && self.have_animation {
            // duration: U32(Val(0), Val(1), Bits(8), Bits(32))
            match self.duration {
                0 => writer.write(2, 0)?,
                1 => writer.write(2, 1)?,
                d if d <= 255 => {
                    writer.write(2, 2)?;
                    writer.write(8, d as u64)?;
                }
                d => {
                    writer.write(2, 3)?;
                    writer.write(32, d as u64)?;
                }
            }
            if self.have_timecodes {
                writer.write(32, self.timecode as u64)?;
            }
        }

        // is_last (for Regular or SkipProgressive)
        if normal_frame {
            writer.write_bit(self.is_last)?;
        }

        // save_as_reference (only when !is_last and not LfFrame)
        if !self.is_last && self.frame_type != FrameType::LfFrame {
            writer.write(2, self.save_as_reference as u64)?;

            // save_before_ct has two independent conditions (libjxl spec):
            // 1. ReferenceOnly frames: ALWAYS present (default true)
            // 2. Normal frames that reset canvas and can be referenced: present (default false)
            if self.frame_type == FrameType::ReferenceOnly {
                writer.write_bit(self.save_before_ct)?;
            } else {
                let full_frame =
                    self.x0 == 0 && self.y0 == 0 && self.width == 0 && self.height == 0;
                let resets_canvas = self.blend_mode == BlendMode::Replace && full_frame;
                let can_be_referenced = self.duration == 0 || self.save_as_reference != 0;
                if resets_canvas && can_be_referenced && normal_frame {
                    writer.write_bit(self.save_before_ct)?;
                }
            }
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
    ///
    /// Crop dimensions use U32(Bits(8), Bits(11)+256, Bits(14)+2048, Bits(30)+18432).
    /// x0/y0 are packed-signed first, then encoded with the same distribution.
    /// Writes crop origin (x0, y0) as UnpackSigned values.
    fn write_crop_origin(&self, writer: &mut BitWriter) -> Result<()> {
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
        Self::write_crop_u32(writer, x0u)?;
        Self::write_crop_u32(writer, y0u)?;
        Ok(())
    }

    /// Encodes a single crop dimension value using U32(Bits(8), Bits(11)+256, Bits(14)+2304, Bits(30)+18688).
    fn write_crop_u32(writer: &mut BitWriter, value: u32) -> Result<()> {
        if value < 256 {
            writer.write(2, 0)?; // selector 0: Bits(8)
            writer.write(8, value as u64)?;
        } else if value < 2304 {
            writer.write(2, 1)?; // selector 1: Bits(11)+256
            writer.write(11, (value - 256) as u64)?;
        } else if value < 18688 {
            writer.write(2, 2)?; // selector 2: Bits(14)+2304
            writer.write(14, (value - 2304) as u64)?;
        } else {
            writer.write(2, 3)?; // selector 3: Bits(30)+18688
            writer.write(30, (value - 18688) as u64)?;
        }
        Ok(())
    }

    /// Writes blending information for the main frame.
    fn write_blending_info(&self, writer: &mut BitWriter) -> Result<()> {
        writer.write_u32_coder(self.blend_mode as u32, 0, 1, 2, 3, 2)?;

        // source: only when not (full_frame && Replace)
        // Full frame is the default (no crop), so source is written for non-Replace modes.
        let full_frame = self.x0 == 0 && self.y0 == 0 && self.width == 0 && self.height == 0;
        if !(full_frame && self.blend_mode == BlendMode::Replace) {
            writer.write(2, self.blend_source as u64)?;
        }

        if self.blend_mode == BlendMode::Blend || self.blend_mode == BlendMode::AlphaWeightedAdd {
            writer.write_u32_coder(self.alpha_blend_channel, 0, 1, 2, 3, 3)?;
            writer.write_bit(false)?; // clamp = false
        }

        Ok(())
    }

    /// Writes blending information for an extra channel.
    fn write_ec_blending_info(&self, mode: BlendMode, writer: &mut BitWriter) -> Result<()> {
        writer.write_u32_coder(mode as u32, 0, 1, 2, 3, 2)?;

        let full_frame = self.x0 == 0 && self.y0 == 0 && self.width == 0 && self.height == 0;
        if !(full_frame && mode == BlendMode::Replace) {
            writer.write(2, 0)?; // source = 0
        }

        if mode == BlendMode::Blend || mode == BlendMode::AlphaWeightedAdd {
            writer.write_u32_coder(0, 0, 1, 2, 3, 3)?; // alpha channel = 0
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

    /// Writes the Passes struct when num_passes > 1.
    ///
    /// Format (from jxl-rs decoder):
    /// - num_ds: u2S(0, 1, 2, Bits(1)+3)
    /// - shift[0..num_passes-1]: Bits(2) each
    /// - downsample[0..num_ds]: u2S(1, 2, 4, 8)
    /// - last_pass[0..num_ds]: u2S(0, 1, 2, Bits(3))
    fn write_passes(&self, writer: &mut BitWriter) -> Result<()> {
        // num_ds: u2S(0, 1, 2, Bits(1)+3)
        writer.write_u32_coder(self.num_ds, 0, 1, 2, 3, 1)?;

        // shift[0..num_passes-1]: Bits(2) each
        for i in 0..self.num_passes.saturating_sub(1) as usize {
            let shift = self.pass_shifts.get(i).copied().unwrap_or(0);
            writer.write(2, shift as u64)?;
        }

        // downsample[0..num_ds]: u2S(1, 2, 4, 8)
        for i in 0..self.num_ds as usize {
            let ds = self.ds_downsample.get(i).copied().unwrap_or(1);
            writer.write_u32_coder(ds, 1, 2, 4, 8, 0)?;
        }

        // last_pass[0..num_ds]: u2S(0, 1, 2, Bits(3))
        for i in 0..self.num_ds as usize {
            let lp = self.ds_last_pass.get(i).copied().unwrap_or(0);
            writer.write_u32_coder(lp, 0, 1, 2, 0, 3)?;
        }

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
            && self.pass_shifts.is_empty()
            && self.x0 == 0
            && self.y0 == 0
            && self.width == 0
            && self.height == 0
            && self.blend_mode == BlendMode::Replace
            && self.blend_source == 0
            && self.save_as_reference == 0
            && !self.save_before_ct
            && self.name.is_empty()
            && !self.have_animation
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
        // Use the dedicated constructor which sets lf_level correctly
        let frame = FrameHeader::lf_frame(32, 32, 1);

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
