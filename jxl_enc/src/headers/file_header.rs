// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! JXL file header (SizeHeader + ImageMetadata).

use crate::bit_writer::BitWriter;
use crate::error::Result;
use crate::JXL_SIGNATURE;

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
        header.metadata.extra_channels.push(ExtraChannelInfo::alpha());
        header
    }

    /// Writes the JXL signature.
    pub fn write_signature(writer: &mut BitWriter) -> Result<()> {
        writer.write_u8(JXL_SIGNATURE[0])?;
        writer.write_u8(JXL_SIGNATURE[1])?;
        Ok(())
    }

    /// Writes the size header.
    fn write_size_header(&self, writer: &mut BitWriter) -> Result<()> {
        // div8 flag: can we represent size as (n+1)*8?
        let div8 = self.height.is_multiple_of(8) && self.width.is_multiple_of(8);
        writer.write_bit(div8)?;

        if div8 {
            let h = self.height / 8 - 1;
            let w = self.width / 8 - 1;
            self.write_size_value(writer, h)?;
            // ratio selector
            let ratio = self.compute_ratio();
            writer.write(3, ratio as u64)?;
            if ratio == 0 {
                self.write_size_value(writer, w)?;
            }
        } else {
            self.write_size_value(writer, self.height - 1)?;
            let ratio = self.compute_ratio();
            writer.write(3, ratio as u64)?;
            if ratio == 0 {
                self.write_size_value(writer, self.width - 1)?;
            }
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

    /// Writes a size value using the JXL size encoding.
    fn write_size_value(&self, writer: &mut BitWriter, value: u32) -> Result<()> {
        if value < 9 {
            writer.write(2, 0)?;
            writer.write(9, value as u64)?;
        } else if value < 9 + (1 << 13) {
            writer.write(2, 1)?;
            writer.write(13, (value - 9) as u64)?;
        } else if value < 9 + (1 << 13) + (1 << 18) {
            writer.write(2, 2)?;
            writer.write(18, (value - 9 - (1 << 13)) as u64)?;
        } else {
            writer.write(2, 3)?;
            writer.write(30, (value - 9 - (1 << 13) - (1 << 18)) as u64)?;
        }
        Ok(())
    }

    /// Writes the complete file header (signature + size + metadata).
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        Self::write_signature(writer)?;
        self.write_size_header(writer)?;
        self.write_image_metadata(writer)?;
        Ok(())
    }

    /// Writes the image metadata.
    fn write_image_metadata(&self, writer: &mut BitWriter) -> Result<()> {
        let meta = &self.metadata;

        // all_default flag
        let all_default = self.is_metadata_default();
        writer.write_bit(all_default)?;

        if all_default {
            return Ok(());
        }

        // extra_fields flag
        let extra_fields = meta.animation.is_some()
            || meta.orientation != Orientation::Identity
            || meta.have_intrinsic_size
            || meta.intensity_target != 255.0;
        writer.write_bit(extra_fields)?;

        if extra_fields {
            // orientation - 1 (3 bits)
            writer.write(3, (meta.orientation as u8 - 1) as u64)?;

            // have_intrinsic_size
            writer.write_bit(meta.have_intrinsic_size)?;
            if meta.have_intrinsic_size {
                self.write_size_value(writer, meta.intrinsic_width)?;
                self.write_size_value(writer, meta.intrinsic_height)?;
            }

            // have_preview (not implemented)
            writer.write_bit(false)?;

            // have_animation
            writer.write_bit(meta.animation.is_some())?;
            if let Some(ref _anim) = meta.animation {
                // TODO: Write animation header
            }
        }

        // bit_depth
        meta.bit_depth.write(writer)?;

        // modular_16_bit_buffer_sufficient (always true for now)
        writer.write_bit(true)?;

        // num_extra_channels
        let num_extra = meta.extra_channels.len() as u32;
        writer.write_u32_coder(num_extra, 0, 1, 2, 1, 12)?;

        for ec in &meta.extra_channels {
            ec.write(writer)?;
        }

        // xyb_encoded (true for lossy, false for lossless)
        writer.write_bit(true)?;

        // color_encoding
        meta.color_encoding.write(writer)?;

        // tone_mapping (not implemented - use default)
        writer.write_bit(true)?; // all_default

        // extensions
        writer.write_bit(false)?; // no extensions

        Ok(())
    }

    /// Checks if all metadata is default.
    fn is_metadata_default(&self) -> bool {
        let meta = &self.metadata;
        meta.bit_depth.bits_per_sample == 8
            && !meta.bit_depth.float_sample
            && meta.extra_channels.is_empty()
            && meta.orientation == Orientation::Identity
            && meta.animation.is_none()
            && !meta.have_intrinsic_size
            && meta.color_encoding.is_srgb()
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
