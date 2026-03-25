// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! JXL file header (SizeHeader + ImageMetadata).

use crate::JXL_SIGNATURE;
use crate::bit_writer::BitWriter;
use crate::error::Result;

use super::color_encoding::ColorEncoding;
use super::extra_channels::ExtraChannelInfo;

/// Orientation of the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Orientation {
    #[default]
    Identity = 1,
    FlipHorizontal = 2,
    Rotate180 = 3,
    FlipVertical = 4,
    Transpose = 5,
    Rotate90CW = 6,
    AntiTranspose = 7,
    Rotate90CCW = 8,
}

/// Bit depth specification.
#[derive(Debug, Clone, Copy)]
pub struct BitDepth {
    /// True if floating point, false if integer.
    pub float_sample: bool,
    /// Bits per sample (for integer) or exponent bits (for float).
    pub bits_per_sample: u32,
    /// Exponent bits for floating point samples.
    pub exponent_bits: u32,
}

impl Default for BitDepth {
    fn default() -> Self {
        Self {
            float_sample: false,
            bits_per_sample: 8,
            exponent_bits: 0,
        }
    }
}

impl BitDepth {
    /// Creates an 8-bit integer depth.
    pub fn uint8() -> Self {
        Self::default()
    }

    /// Creates a 16-bit integer depth.
    pub fn uint16() -> Self {
        Self {
            float_sample: false,
            bits_per_sample: 16,
            exponent_bits: 0,
        }
    }

    /// Creates a 32-bit float depth.
    pub fn float32() -> Self {
        Self {
            float_sample: true,
            bits_per_sample: 32,
            exponent_bits: 8,
        }
    }

    /// Creates a 16-bit half-float depth.
    pub fn float16() -> Self {
        Self {
            float_sample: true,
            bits_per_sample: 16,
            exponent_bits: 5,
        }
    }
}

/// Animation parameters.
#[derive(Debug, Clone, Default)]
pub struct AnimationHeader {
    /// Ticks per second numerator.
    pub tps_numerator: u32,
    /// Ticks per second denominator.
    pub tps_denominator: u32,
    /// Number of loops (0 = infinite).
    pub num_loops: u32,
    /// Whether frames have varying durations.
    pub have_timecodes: bool,
}

impl AnimationHeader {
    /// Writes the AnimationHeader to the bitstream.
    ///
    /// Matches libjxl's `AnimationHeader::VisitFields`:
    /// - tps_numerator: u2S(100, 1000, Bits(10)+1, Bits(30)+1)
    /// - tps_denominator: u2S(1, 1001, Bits(8)+1, Bits(10)+1)
    /// - num_loops: u2S(0, Bits(3), Bits(16), Bits(32))
    /// - have_timecodes: Bool(false)
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        // tps_numerator: u2S(100, 1000, BitsOffset(10,1), BitsOffset(30,1))
        match self.tps_numerator {
            100 => writer.write(2, 0)?,
            1000 => writer.write(2, 1)?,
            v if (1..=1024).contains(&v) => {
                writer.write(2, 2)?;
                writer.write(10, (v - 1) as u64)?;
            }
            v => {
                debug_assert!(v >= 1, "tps_numerator must be >= 1");
                writer.write(2, 3)?;
                writer.write(30, (v - 1) as u64)?;
            }
        }

        // tps_denominator: u2S(1, 1001, BitsOffset(8,1), BitsOffset(10,1))
        match self.tps_denominator {
            1 => writer.write(2, 0)?,
            1001 => writer.write(2, 1)?,
            v @ 2..=256 => {
                writer.write(2, 2)?;
                writer.write(8, (v - 1) as u64)?;
            }
            v => {
                debug_assert!((1..=1025).contains(&v), "tps_denominator {v} out of range");
                writer.write(2, 3)?;
                writer.write(10, (v - 1) as u64)?;
            }
        }

        // num_loops: u2S(0, Bits(3), Bits(16), Bits(32))
        match self.num_loops {
            0 => writer.write(2, 0)?,
            v @ 1..=7 => {
                writer.write(2, 1)?;
                writer.write(3, v as u64)?;
            }
            v @ 8..=65535 => {
                writer.write(2, 2)?;
                writer.write(16, v as u64)?;
            }
            v => {
                writer.write(2, 3)?;
                writer.write(32, v as u64)?;
            }
        }

        // have_timecodes: Bool(default=false)
        writer.write_bit(self.have_timecodes)?;

        Ok(())
    }
}

/// Image metadata that appears once per file.
#[derive(Debug, Clone)]
pub struct ImageMetadata {
    /// Bit depth configuration.
    pub bit_depth: BitDepth,
    /// Color encoding (color space, transfer function, etc.).
    pub color_encoding: ColorEncoding,
    /// Extra channels (alpha, depth, etc.).
    pub extra_channels: Vec<ExtraChannelInfo>,
    /// Image orientation.
    pub orientation: Orientation,
    /// Animation parameters (None if not animated).
    pub animation: Option<AnimationHeader>,
    /// Intensity target for HDR in nits.
    pub intensity_target: f32,
    /// Minimum nits for tone mapping.
    pub min_nits: f32,
    /// Whether intrinsic size differs from coded size.
    pub have_intrinsic_size: bool,
    /// Intrinsic width (if have_intrinsic_size).
    pub intrinsic_width: u32,
    /// Intrinsic height (if have_intrinsic_size).
    pub intrinsic_height: u32,
    /// Whether image uses XYB color encoding (true for lossy, false for lossless).
    pub xyb_encoded: bool,
}

impl Default for ImageMetadata {
    fn default() -> Self {
        Self {
            bit_depth: BitDepth::default(),
            color_encoding: ColorEncoding::default(),
            extra_channels: Vec::new(),
            orientation: Orientation::default(),
            animation: None,
            intensity_target: 255.0,
            min_nits: 0.0,
            have_intrinsic_size: false,
            intrinsic_width: 0,
            intrinsic_height: 0,
            xyb_encoded: false, // Default to lossless (non-XYB)
        }
    }
}

/// Complete JXL file header.
#[derive(Debug, Clone)]
pub struct FileHeader {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Image metadata.
    pub metadata: ImageMetadata,
}

impl FileHeader {
    /// Creates a new file header for an RGB image.
    pub fn new_rgb(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            metadata: ImageMetadata::default(),
        }
    }

    /// Creates a new file header for an RGBA image.
    pub fn new_rgba(width: u32, height: u32) -> Self {
        let mut header = Self::new_rgb(width, height);
        header
            .metadata
            .extra_channels
            .push(ExtraChannelInfo::alpha());
        header
    }

    /// Creates a new file header for a grayscale image.
    pub fn new_gray(width: u32, height: u32) -> Self {
        let mut header = Self::new_rgb(width, height);
        header.metadata.color_encoding = ColorEncoding::gray();
        header
    }

    /// Creates a new file header for a lossy RGB image (VarDCT/XYB encoded).
    pub fn new_rgb_lossy(width: u32, height: u32) -> Self {
        let mut header = Self::new_rgb(width, height);
        header.metadata.xyb_encoded = true;
        header
    }

    /// Writes the JXL signature.
    pub fn write_signature(writer: &mut BitWriter) -> Result<()> {
        writer.write_u8(JXL_SIGNATURE[0])?;
        writer.write_u8(JXL_SIGNATURE[1])?;
        Ok(())
    }

    /// Writes the size header.
    ///
    /// JXL Size format:
    /// - small: Bool (1 bit) - true if both dimensions are multiples of 8 and <= 256
    /// - If small:
    ///   - ysize_div8: Bits(5) + 1 (height/8, range 1-32)
    ///   - ratio: Bits(3)
    ///   - If ratio == 0: xsize_div8: Bits(5) + 1 (width/8, range 1-32)
    /// - If !small:
    ///   - ysize: 1 + u2S(Bits(9), Bits(13), Bits(18), Bits(30))
    ///   - ratio: Bits(3)
    ///   - If ratio == 0: xsize: 1 + u2S(Bits(9), Bits(13), Bits(18), Bits(30))
    fn write_size_header(&self, writer: &mut BitWriter) -> Result<()> {
        // small = true if both dimensions are multiples of 8 and fit in 5 bits (8-256)
        let h_div8 = self.height.is_multiple_of(8) && self.height / 8 >= 1 && self.height / 8 <= 32;
        let w_div8 = self.width.is_multiple_of(8) && self.width / 8 >= 1 && self.width / 8 <= 32;
        let small = h_div8 && w_div8;

        crate::trace::debug_eprintln!(
            "SIZE_HDR: {}x{}, small={}, h_div8={}, w_div8={}",
            self.width,
            self.height,
            small,
            h_div8,
            w_div8
        );
        writer.write_bit(small)?;

        if small {
            // ysize_div8_minus_1: Bits(5), decoder adds 1 then multiplies by 8
            crate::trace::debug_eprintln!("SIZE_HDR: ysize_div8_minus_1 = {}", self.height / 8 - 1);
            writer.write(5, (self.height / 8 - 1) as u64)?;

            let ratio = self.compute_ratio();
            crate::trace::debug_eprintln!("SIZE_HDR: ratio = {}", ratio);
            writer.write(3, ratio as u64)?;

            if ratio == 0 {
                // xsize_div8_minus_1: Bits(5), decoder adds 1 then multiplies by 8
                crate::trace::debug_eprintln!(
                    "SIZE_HDR: xsize_div8_minus_1 = {}",
                    self.width / 8 - 1
                );
                writer.write(5, (self.width / 8 - 1) as u64)?;
            }
        } else {
            // ysize: 1 + u2S(Bits(9), Bits(13), Bits(18), Bits(30))
            // Write height - 1 using u2S encoding
            self.write_size_u2s(writer, self.height - 1)?;

            let ratio = self.compute_ratio();
            writer.write(3, ratio as u64)?;

            if ratio == 0 {
                // xsize: 1 + u2S(Bits(9), Bits(13), Bits(18), Bits(30))
                self.write_size_u2s(writer, self.width - 1)?;
            }
        }

        Ok(())
    }

    /// Writes a size value using u2S(Bits(9), Bits(13), Bits(18), Bits(30)) encoding.
    /// The decoder adds 1 to the result, so we write value directly (not value-1).
    fn write_size_u2s(&self, writer: &mut BitWriter, value: u32) -> Result<()> {
        if value < (1 << 9) {
            writer.write(2, 0)?; // selector 0
            writer.write(9, value as u64)?;
        } else if value < (1 << 13) {
            writer.write(2, 1)?; // selector 1
            writer.write(13, value as u64)?;
        } else if value < (1 << 18) {
            writer.write(2, 2)?; // selector 2
            writer.write(18, value as u64)?;
        } else {
            writer.write(2, 3)?; // selector 3
            writer.write(30, value as u64)?;
        }
        Ok(())
    }

    /// Computes the aspect ratio selector (0 = explicit width).
    fn compute_ratio(&self) -> u8 {
        // Ratio selectors: 1=1:1, 2=12:10, 3=4:3, 4=3:2, 5=16:9, 6=5:4, 7=2:1
        if self.width == self.height {
            1 // 1:1
        } else if self.width * 10 == self.height * 12 {
            2 // 12:10
        } else if self.width * 3 == self.height * 4 {
            3 // 4:3
        } else if self.width * 2 == self.height * 3 {
            4 // 3:2
        } else if self.width * 9 == self.height * 16 {
            5 // 16:9
        } else if self.width * 4 == self.height * 5 {
            6 // 5:4
        } else if self.width == self.height * 2 {
            7 // 2:1
        } else {
            0 // Explicit
        }
    }

    /// Writes the complete file header (signature + size + metadata + transform_data).
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        crate::trace::debug_eprintln!("FHDR [bit {}]: Starting file header", writer.bits_written());
        Self::write_signature(writer)?;
        crate::trace::debug_eprintln!("FHDR [bit {}]: After signature", writer.bits_written());
        self.write_size_header(writer)?;
        crate::trace::debug_eprintln!("FHDR [bit {}]: After size header", writer.bits_written());
        self.write_image_metadata(writer)?;
        crate::trace::debug_eprintln!("FHDR [bit {}]: After metadata", writer.bits_written());
        // CustomTransformData - written after ImageMetadata
        // For simple images, all_default = true (just 1 bit)
        self.write_transform_data(writer)?;
        crate::trace::debug_eprintln!("FHDR [bit {}]: After transform_data", writer.bits_written());
        Ok(())
    }

    /// Writes the CustomTransformData bundle.
    /// For basic encoding (no custom transform settings), this is just all_default=true (1 bit).
    fn write_transform_data(&self, writer: &mut BitWriter) -> Result<()> {
        // CustomTransformData.all_default = true
        // This is the default case - no custom upsampling weights or opsin matrix
        crate::trace::debug_eprintln!(
            "XFRM [bit {}]: transform_data.all_default = true",
            writer.bits_written()
        );
        writer.write_bit(true)?;
        Ok(())
    }

    /// Writes the image metadata.
    fn write_image_metadata(&self, writer: &mut BitWriter) -> Result<()> {
        let meta = &self.metadata;

        // all_default flag
        let all_default = self.is_metadata_default();
        crate::trace::debug_eprintln!(
            "META [bit {}]: all_default = {}",
            writer.bits_written(),
            all_default
        );
        writer.write_bit(all_default)?;

        if all_default {
            return Ok(());
        }

        // extra_fields flag
        let extra_fields = meta.animation.is_some()
            || meta.orientation != Orientation::Identity
            || meta.have_intrinsic_size
            || meta.intensity_target != 255.0
            || meta.min_nits != 0.0;
        crate::trace::debug_eprintln!(
            "META [bit {}]: extra_fields = {}",
            writer.bits_written(),
            extra_fields
        );
        writer.write_bit(extra_fields)?;

        if extra_fields {
            // orientation - 1 (3 bits)
            writer.write(3, (meta.orientation as u8 - 1) as u64)?;

            // have_intrinsic_size
            writer.write_bit(meta.have_intrinsic_size)?;
            if meta.have_intrinsic_size {
                // Intrinsic size uses same u2S encoding as Size
                self.write_size_u2s(writer, meta.intrinsic_width - 1)?;
                self.write_size_u2s(writer, meta.intrinsic_height - 1)?;
            }

            // have_preview (not implemented)
            writer.write_bit(false)?;

            // have_animation
            writer.write_bit(meta.animation.is_some())?;
            if let Some(ref anim) = meta.animation {
                anim.write(writer)?;
            }
        }

        // bit_depth
        crate::trace::debug_eprintln!("META [bit {}]: Writing bit_depth", writer.bits_written());
        meta.bit_depth.write(writer)?;
        crate::trace::debug_eprintln!("META [bit {}]: After bit_depth", writer.bits_written());

        // modular_16_bit_buffer_sufficient
        // Default is true for bit depths <= 12
        let mod16_sufficient = meta.bit_depth.bits_per_sample <= 12;
        crate::trace::debug_eprintln!(
            "META [bit {}]: modular_16_bit_buffer_sufficient = {}",
            writer.bits_written(),
            mod16_sufficient
        );
        writer.write_bit(mod16_sufficient)?;

        // num_extra_channels
        let num_extra = meta.extra_channels.len() as u32;
        crate::trace::debug_eprintln!(
            "META [bit {}]: num_extra_channels = {}",
            writer.bits_written(),
            num_extra
        );
        writer.write_u32_coder(num_extra, 0, 1, 2, 1, 12)?;

        for ec in &meta.extra_channels {
            ec.write(writer)?;
        }

        // xyb_encoded (true for lossy, false for lossless)
        crate::trace::debug_eprintln!(
            "META [bit {}]: xyb_encoded = {}",
            writer.bits_written(),
            meta.xyb_encoded
        );
        writer.write_bit(meta.xyb_encoded)?;

        // color_encoding
        crate::trace::debug_eprintln!(
            "META [bit {}]: Writing color_encoding",
            writer.bits_written()
        );
        meta.color_encoding.write(writer)?;
        crate::trace::debug_eprintln!("META [bit {}]: After color_encoding", writer.bits_written());

        // tone_mapping - only if extra_fields
        if extra_fields {
            let tone_all_default = meta.intensity_target == 255.0 && meta.min_nits == 0.0;
            writer.write_bit(tone_all_default)?;
            if !tone_all_default {
                crate::f16::write_f16(meta.intensity_target, writer)?;
                crate::f16::write_f16(meta.min_nits, writer)?;
                writer.write_bit(false)?; // relative_to_max_display
                crate::f16::write_f16(0.0, writer)?; // linear_below
            }
        }

        // extensions (u64 selector, 0 = no extensions)
        // u64 encoding: 2-bit selector, 0 means value 0
        writer.write(2, 0)?;

        Ok(())
    }

    /// Checks if all metadata is default.
    /// Per JXL spec, all_default=true implies xyb_encoded=false (lossless mode).
    fn is_metadata_default(&self) -> bool {
        // For now, always return false to write explicit metadata.
        // This ensures compatibility while we investigate the all_default parsing issue.
        // TODO: Enable all_default optimization once we confirm decoder compatibility.
        false
    }
}

impl BitDepth {
    /// Writes the bit depth to the bitstream.
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        writer.write_bit(self.float_sample)?;
        if self.float_sample {
            // bits_per_sample for float: u2S(32, 16, 24, 1 + Bits(6))
            writer.write_u32_coder(self.bits_per_sample, 32, 16, 24, 1, 6)?;
            // exponent_bits: 1 + Bits(4)
            writer.write(4, (self.exponent_bits - 1) as u64)?;
        } else {
            // bits_per_sample for int: u2S(8, 10, 12, 1 + Bits(6))
            writer.write_u32_coder(self.bits_per_sample, 8, 10, 12, 1, 6)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signature() {
        let mut writer = BitWriter::new();
        FileHeader::write_signature(&mut writer).unwrap();
        let bytes = writer.finish();
        assert_eq!(bytes, vec![0xFF, 0x0A]);
    }

    #[test]
    fn test_simple_header() {
        let header = FileHeader::new_rgb(256, 256);
        let mut writer = BitWriter::new();
        header.write(&mut writer).unwrap();

        let bytes = writer.finish_with_padding();
        // Should start with JXL signature
        assert_eq!(&bytes[0..2], &[0xFF, 0x0A]);
    }
}
