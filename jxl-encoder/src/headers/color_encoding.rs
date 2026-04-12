// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

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
    /// Perceptual (libjxl default for lossless encoding).
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
    /// Custom gamma (encoding exponent). When Some, writes have_gamma=true + 24-bit value.
    /// Example: 0.45455 for standard gamma 2.2 (display gamma = 1/0.45455 ≈ 2.2).
    pub gamma: Option<f32>,
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
            gamma: None,
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
            gamma: None,
        }
    }

    /// Creates a grayscale sRGB color encoding.
    pub fn gray() -> Self {
        Self {
            color_space: ColorSpace::Gray,
            white_point: WhitePoint::D65,
            primaries: Primaries::Srgb,
            transfer_function: TransferFunction::Srgb,
            rendering_intent: RenderingIntent::Perceptual,
            want_icc: false,
            gamma: None,
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
            gamma: None,
        }
    }

    /// Creates an sRGB color encoding with a custom gamma transfer function.
    ///
    /// Used for PNGs with `gAMA` chunk but no `sRGB` chunk. The gamma value
    /// is the encoding exponent (e.g., 0.45455 for standard gamma 2.2).
    pub fn with_gamma(gamma: f32) -> Self {
        Self {
            gamma: Some(gamma),
            ..Self::srgb()
        }
    }

    /// Creates a grayscale color encoding with a custom gamma transfer function.
    pub fn gray_with_gamma(gamma: f32) -> Self {
        Self {
            gamma: Some(gamma),
            ..Self::gray()
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
            gamma: None,
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
            gamma: None,
        }
    }

    /// Returns true if this matches the JXL default color encoding.
    /// (sRGB with Perceptual rendering intent, no ICC)
    ///
    /// When all_default=true for metadata with xyb_encoded=true (lossy mode),
    /// the decoder assumes sRGB input color space.
    pub fn is_srgb(&self) -> bool {
        self.color_space == ColorSpace::Rgb
            && self.white_point == WhitePoint::D65
            && self.primaries == Primaries::Srgb
            && self.transfer_function == TransferFunction::Srgb
            && self.rendering_intent == RenderingIntent::Perceptual
            && !self.want_icc
            && self.gamma.is_none()
    }

    /// Returns true if this is grayscale.
    pub fn is_gray(&self) -> bool {
        self.color_space == ColorSpace::Gray
    }

    /// Writes the color encoding to the bitstream.
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        // all_default flag
        let all_default = self.is_srgb();
        crate::trace::debug_eprintln!(
            "CENC [bit {}]: all_default = {}",
            writer.bits_written(),
            all_default
        );
        writer.write_bit(all_default)?;

        if all_default {
            return Ok(());
        }

        // want_icc
        crate::trace::debug_eprintln!(
            "CENC [bit {}]: want_icc = {}",
            writer.bits_written(),
            self.want_icc
        );
        writer.write_bit(self.want_icc)?;

        // color_space is ALWAYS written (even when want_icc=true, it affects decoding)
        crate::trace::debug_eprintln!(
            "CENC [bit {}]: color_space = {:?} ({})",
            writer.bits_written(),
            self.color_space,
            self.color_space as u8
        );
        writer.write(2, self.color_space as u64)?;

        if self.want_icc {
            // When want_icc=true, white point/primaries/transfer/rendering intent are not written
            return Ok(());
        }

        // white_point - uses jxl-rs default u2S(0, 1, Bits(4)+2, Bits(6)+18)
        let wp = match self.white_point {
            WhitePoint::D65 => 1,
            WhitePoint::Custom => 2,
            WhitePoint::E => 10,
            WhitePoint::Dci => 11,
        };
        crate::trace::debug_eprintln!(
            "CENC [bit {}]: white_point = {:?} ({})",
            writer.bits_written(),
            self.white_point,
            wp
        );
        writer.write_enum_default(wp)?;
        if self.white_point == WhitePoint::Custom {
            return Err(crate::error::Error::NotImplemented(
                "custom white point encoding".into(),
            ));
        }

        // primaries (only for RGB) - uses jxl-rs default u2S encoding
        if self.color_space == ColorSpace::Rgb {
            let prim = match self.primaries {
                Primaries::Srgb => 1,
                Primaries::Custom => 2,
                Primaries::Bt2100 => 9,
                Primaries::P3 => 11,
            };
            crate::trace::debug_eprintln!(
                "CENC [bit {}]: primaries = {:?} ({})",
                writer.bits_written(),
                self.primaries,
                prim
            );
            writer.write_enum_default(prim)?;
            if self.primaries == Primaries::Custom {
                return Err(crate::error::Error::NotImplemented(
                    "custom primaries encoding".into(),
                ));
            }
        } else {
            crate::trace::debug_eprintln!(
                "CENC [bit {}]: primaries skipped (not RGB)",
                writer.bits_written()
            );
        }

        // have_gamma
        let have_gamma = self.gamma.is_some();
        crate::trace::debug_eprintln!(
            "CENC [bit {}]: have_gamma = {}",
            writer.bits_written(),
            have_gamma
        );
        writer.write_bit(have_gamma)?;

        if have_gamma {
            let g = self.gamma.expect("gamma must be set when have_gamma=true");
            // JXL spec: 24-bit integer = round(gamma * 10_000_000), clamped to [1, 2^24-1]
            let encoded = (g * 10_000_000.0).round() as u32;
            crate::trace::debug_eprintln!(
                "CENC [bit {}]: gamma = {} (encoded {})",
                writer.bits_written(),
                g,
                encoded
            );
            writer.write(24, encoded as u64)?;
        } else {
            // transfer_function - uses jxl-rs default u2S encoding
            let tf = match self.transfer_function {
                TransferFunction::Bt709 => 1,
                TransferFunction::Unknown => 2,
                TransferFunction::Linear => 8,
                TransferFunction::Srgb => 13,
                TransferFunction::Pq => 16,
                TransferFunction::Dci => 17,
                TransferFunction::Hlg => 18,
            };
            crate::trace::debug_eprintln!(
                "CENC [bit {}]: transfer_function = {:?} ({})",
                writer.bits_written(),
                self.transfer_function,
                tf
            );
            writer.write_enum_default(tf)?;
        }

        // rendering_intent
        crate::trace::debug_eprintln!(
            "CENC [bit {}]: rendering_intent = {:?} ({})",
            writer.bits_written(),
            self.rendering_intent,
            self.rendering_intent as u8
        );
        writer.write(2, self.rendering_intent as u64)?;
        crate::trace::debug_eprintln!("CENC [bit {}]: color_encoding done", writer.bits_written());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_is_default() {
        let enc = ColorEncoding::srgb();
        // is_srgb() returns true for default sRGB encoding
        // (enables all_default=true for metadata in XYB mode)
        assert!(enc.is_srgb());
    }

    #[test]
    fn test_write_srgb() {
        let enc = ColorEncoding::srgb();
        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        writer.zero_pad_to_byte();

        // With is_srgb() returning true, all_default=true is written (1 bit)
        // Padded to byte boundary = 8 bits
        assert_eq!(writer.bits_written(), 8);
    }

    #[test]
    fn test_write_non_default_srgb() {
        // Non-default sRGB (Relative intent instead of Perceptual)
        let enc = ColorEncoding {
            color_space: ColorSpace::Rgb,
            white_point: WhitePoint::D65,
            primaries: Primaries::Srgb,
            transfer_function: TransferFunction::Srgb,
            rendering_intent: RenderingIntent::Relative, // Non-default
            want_icc: false,
            gamma: None,
        };
        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        writer.zero_pad_to_byte();

        // With is_srgb() returning false (Relative != Perceptual),
        // explicit color encoding is written:
        // all_default=0 (1), want_icc=0 (1), color_space=0 (2),
        // white_point D65=1 (2), primaries sRGB=1 (2), have_gamma=0 (1),
        // transfer_function sRGB=13 (2+4=6), rendering_intent=1 (2)
        // Total: 17 bits -> 24 bits padded = 3 bytes
        assert_eq!(writer.bits_written(), 24);
    }

    #[test]
    fn test_color_space_values() {
        assert_eq!(ColorSpace::Rgb as u8, 0);
        assert_eq!(ColorSpace::Gray as u8, 1);
        assert_eq!(ColorSpace::Xyb as u8, 2);
        assert_eq!(ColorSpace::Unknown as u8, 3);
    }

    #[test]
    fn test_white_point_values() {
        assert_eq!(WhitePoint::D65 as u8, 1);
        assert_eq!(WhitePoint::Custom as u8, 2);
        assert_eq!(WhitePoint::E as u8, 10);
        assert_eq!(WhitePoint::Dci as u8, 11);
    }

    #[test]
    fn test_primaries_values() {
        assert_eq!(Primaries::Srgb as u8, 1);
        assert_eq!(Primaries::Custom as u8, 2);
        assert_eq!(Primaries::Bt2100 as u8, 9);
        assert_eq!(Primaries::P3 as u8, 11);
    }

    #[test]
    fn test_transfer_function_values() {
        assert_eq!(TransferFunction::Bt709 as u8, 1);
        assert_eq!(TransferFunction::Unknown as u8, 2);
        assert_eq!(TransferFunction::Linear as u8, 8);
        assert_eq!(TransferFunction::Srgb as u8, 13);
        assert_eq!(TransferFunction::Pq as u8, 16);
        assert_eq!(TransferFunction::Dci as u8, 17);
        assert_eq!(TransferFunction::Hlg as u8, 18);
    }

    #[test]
    fn test_rendering_intent_values() {
        assert_eq!(RenderingIntent::Perceptual as u8, 0);
        assert_eq!(RenderingIntent::Relative as u8, 1);
        assert_eq!(RenderingIntent::Saturation as u8, 2);
        assert_eq!(RenderingIntent::Absolute as u8, 3);
    }

    #[test]
    fn test_write_linear_srgb() {
        let enc = ColorEncoding::linear_srgb();
        assert_eq!(enc.transfer_function, TransferFunction::Linear);

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_grayscale() {
        let enc = ColorEncoding::grayscale();
        assert!(enc.is_gray());
        assert_eq!(enc.color_space, ColorSpace::Gray);

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        // Grayscale doesn't write primaries
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_gray() {
        let enc = ColorEncoding::gray();
        assert!(enc.is_gray());

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_display_p3() {
        let enc = ColorEncoding::display_p3();
        assert_eq!(enc.primaries, Primaries::P3);

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_bt2100_pq() {
        let enc = ColorEncoding::bt2100_pq();
        assert_eq!(enc.primaries, Primaries::Bt2100);
        assert_eq!(enc.transfer_function, TransferFunction::Pq);

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_with_want_icc() {
        let mut enc = ColorEncoding::srgb();
        enc.want_icc = true;

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        // With want_icc=true: all_default=0 (1), want_icc=1 (1), color_space (2) = 4 bits
        assert_eq!(writer.bits_written(), 4);
    }

    #[test]
    fn test_write_bt709_transfer() {
        let mut enc = ColorEncoding::srgb();
        enc.transfer_function = TransferFunction::Bt709;

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_dci_transfer() {
        let mut enc = ColorEncoding::srgb();
        enc.transfer_function = TransferFunction::Dci;

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_hlg_transfer() {
        let mut enc = ColorEncoding::srgb();
        enc.transfer_function = TransferFunction::Hlg;

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_e_white_point() {
        let mut enc = ColorEncoding::srgb();
        enc.white_point = WhitePoint::E;

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_dci_white_point() {
        let mut enc = ColorEncoding::srgb();
        enc.white_point = WhitePoint::Dci;

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_rendering_intent_saturation() {
        let mut enc = ColorEncoding::srgb();
        enc.rendering_intent = RenderingIntent::Saturation;

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_rendering_intent_absolute() {
        let mut enc = ColorEncoding::srgb();
        enc.rendering_intent = RenderingIntent::Absolute;

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_xyb_color_space() {
        let mut enc = ColorEncoding::srgb();
        enc.color_space = ColorSpace::Xyb;

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        // XYB doesn't write primaries (not RGB)
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_unknown_color_space() {
        let mut enc = ColorEncoding::srgb();
        enc.color_space = ColorSpace::Unknown;

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        // Unknown color space doesn't write primaries
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_default_encoding() {
        let enc = ColorEncoding::default();
        assert_eq!(enc.color_space, ColorSpace::Rgb);
        assert_eq!(enc.white_point, WhitePoint::D65);
        assert_eq!(enc.primaries, Primaries::Srgb);
        assert_eq!(enc.transfer_function, TransferFunction::Srgb);
        assert_eq!(enc.rendering_intent, RenderingIntent::Perceptual);
        assert!(!enc.want_icc);
        assert!(enc.gamma.is_none());
    }

    #[test]
    fn test_gamma_encoding() {
        // Standard gamma 2.2: encoding exponent = 0.45455
        let enc = ColorEncoding::with_gamma(0.45455);
        assert!(!enc.is_srgb()); // gamma set → not sRGB default
        assert_eq!(enc.gamma, Some(0.45455));

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        writer.zero_pad_to_byte();

        // Verify: 0.45455 * 10_000_000 = 4_545_500
        let encoded = (0.45455_f32 * 10_000_000.0).round() as u32;
        assert_eq!(encoded, 4_545_500);

        // Encoding should be longer than sRGB default (1 bit)
        // all_default=0(1) + want_icc=0(1) + color_space=0(2) + white_point=1(2) +
        // primaries=1(2) + have_gamma=1(1) + gamma(24) + rendering_intent=0(2) = 35 bits
        assert_eq!(writer.bits_written(), 40); // 35 bits padded to 5 bytes
    }

    #[test]
    fn test_gray_with_gamma() {
        let enc = ColorEncoding::gray_with_gamma(0.45455);
        assert!(enc.is_gray());
        assert_eq!(enc.gamma, Some(0.45455));
        assert!(!enc.is_srgb());

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        // Should write without error (grayscale skips primaries)
        assert!(writer.bits_written() > 0);
    }
}
