// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Color encoding structures for JPEG XL.

use crate::bit_writer::BitWriter;
use crate::error::Result;

/// Color space enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ColorSpace {
    /// RGB color space.
    #[default]
    Rgb = 0,
    /// Grayscale.
    Gray = 1,
    /// XYB (perceptual color space used internally by JXL).
    Xyb = 2,
    /// Unknown/custom color space.
    Unknown = 3,
}

/// White point enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum WhitePoint {
    /// D65 white point (sRGB, Display P3).
    #[default]
    D65 = 1,
    /// Custom white point.
    Custom = 2,
    /// E white point.
    E = 10,
    /// DCI white point.
    Dci = 11,
}

/// Primaries enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Primaries {
    /// sRGB primaries.
    #[default]
    Srgb = 1,
    /// Custom primaries.
    Custom = 2,
    /// BT.2100 primaries.
    Bt2100 = 9,
    /// P3 primaries.
    P3 = 11,
}

/// Transfer function enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TransferFunction {
    /// BT.709 transfer function.
    Bt709 = 1,
    /// Unknown transfer function.
    Unknown = 2,
    /// Linear (gamma 1.0).
    Linear = 8,
    /// sRGB transfer function.
    #[default]
    Srgb = 13,
    /// PQ (Perceptual Quantizer) for HDR.
    Pq = 16,
    /// DCI gamma (2.6).
    Dci = 17,
    /// HLG (Hybrid Log-Gamma) for HDR.
    Hlg = 18,
}

/// Rendering intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum RenderingIntent {
    /// Perceptual.
    #[default]
    Perceptual = 0,
    /// Relative colorimetric.
    Relative = 1,
    /// Saturation.
    Saturation = 2,
    /// Absolute colorimetric.
    Absolute = 3,
}

/// Complete color encoding specification.
#[derive(Debug, Clone, Default)]
pub struct ColorEncoding {
    /// Color space.
    pub color_space: ColorSpace,
    /// White point.
    pub white_point: WhitePoint,
    /// Primaries (for RGB).
    pub primaries: Primaries,
    /// Transfer function.
    pub transfer_function: TransferFunction,
    /// Rendering intent.
    pub rendering_intent: RenderingIntent,
    /// Whether this uses an ICC profile.
    pub want_icc: bool,
}

impl ColorEncoding {
    /// Creates a standard sRGB color encoding.
    pub fn srgb() -> Self {
        Self {
            color_space: ColorSpace::Rgb,
            white_point: WhitePoint::D65,
            primaries: Primaries::Srgb,
            transfer_function: TransferFunction::Srgb,
            rendering_intent: RenderingIntent::Perceptual,
            want_icc: false,
        }
    }

    /// Creates a linear sRGB color encoding.
    pub fn linear_srgb() -> Self {
        Self {
            color_space: ColorSpace::Rgb,
            white_point: WhitePoint::D65,
            primaries: Primaries::Srgb,
            transfer_function: TransferFunction::Linear,
            rendering_intent: RenderingIntent::Perceptual,
            want_icc: false,
        }
    }

    /// Creates a Display P3 color encoding.
    pub fn display_p3() -> Self {
        Self {
            color_space: ColorSpace::Rgb,
            white_point: WhitePoint::D65,
            primaries: Primaries::P3,
            transfer_function: TransferFunction::Srgb,
            rendering_intent: RenderingIntent::Perceptual,
            want_icc: false,
        }
    }

    /// Creates a BT.2100 PQ (HDR) color encoding.
    pub fn bt2100_pq() -> Self {
        Self {
            color_space: ColorSpace::Rgb,
            white_point: WhitePoint::D65,
            primaries: Primaries::Bt2100,
            transfer_function: TransferFunction::Pq,
            rendering_intent: RenderingIntent::Perceptual,
            want_icc: false,
        }
    }

    /// Creates a grayscale color encoding.
    pub fn grayscale() -> Self {
        Self {
            color_space: ColorSpace::Gray,
            white_point: WhitePoint::D65,
            primaries: Primaries::Srgb,
            transfer_function: TransferFunction::Srgb,
            rendering_intent: RenderingIntent::Perceptual,
            want_icc: false,
        }
    }

    /// Returns true if this is standard sRGB.
    pub fn is_srgb(&self) -> bool {
        self.color_space == ColorSpace::Rgb
            && self.white_point == WhitePoint::D65
            && self.primaries == Primaries::Srgb
            && self.transfer_function == TransferFunction::Srgb
            && !self.want_icc
    }

    /// Returns true if this is grayscale.
    pub fn is_gray(&self) -> bool {
        self.color_space == ColorSpace::Gray
    }

    /// Writes the color encoding to the bitstream.
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        // all_default flag
        let all_default = self.is_srgb();
        writer.write_bit(all_default)?;

        if all_default {
            return Ok(());
        }

        // want_icc
        writer.write_bit(self.want_icc)?;

        if self.want_icc {
            // ICC profile data would follow (not implemented)
            return Ok(());
        }

        // color_space
        writer.write(2, self.color_space as u64)?;

        // white_point
        let wp = match self.white_point {
            WhitePoint::D65 => 1,
            WhitePoint::Custom => 2,
            WhitePoint::E => 10,
            WhitePoint::Dci => 11,
        };
        writer.write_u32_coder(wp, 1, 2, 10, 1, 4)?;
        if self.white_point == WhitePoint::Custom {
            // Custom white point coordinates would follow
            todo!("Custom white point not implemented");
        }

        // primaries (only for RGB)
        if self.color_space == ColorSpace::Rgb {
            let prim = match self.primaries {
                Primaries::Srgb => 1,
                Primaries::Custom => 2,
                Primaries::Bt2100 => 9,
                Primaries::P3 => 11,
            };
            writer.write_u32_coder(prim, 1, 2, 9, 1, 4)?;
            if self.primaries == Primaries::Custom {
                // Custom primaries would follow
                todo!("Custom primaries not implemented");
            }
        }

        // have_gamma
        let have_gamma = self.transfer_function == TransferFunction::Unknown;
        writer.write_bit(have_gamma)?;

        if have_gamma {
            // Custom gamma would follow
            todo!("Custom gamma not implemented");
        } else {
            // transfer_function
            let tf = match self.transfer_function {
                TransferFunction::Bt709 => 1,
                TransferFunction::Unknown => 2,
                TransferFunction::Linear => 8,
                TransferFunction::Srgb => 13,
                TransferFunction::Pq => 16,
                TransferFunction::Dci => 17,
                TransferFunction::Hlg => 18,
            };
            writer.write_u32_coder(tf, 1, 8, 13, 1, 5)?;
        }

        // rendering_intent
        writer.write(2, self.rendering_intent as u64)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_is_default() {
        let enc = ColorEncoding::srgb();
        assert!(enc.is_srgb());
    }

    #[test]
    fn test_write_srgb() {
        let enc = ColorEncoding::srgb();
        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        writer.zero_pad_to_byte();

        // sRGB should just write a single 1 bit (all_default=true)
        assert_eq!(writer.bits_written(), 8); // Padded to byte
    }
}
