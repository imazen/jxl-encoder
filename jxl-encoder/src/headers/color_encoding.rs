// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Color encoding structures for JPEG XL.

use crate::bit_writer::BitWriter;
use crate::error::{Error, Result};

/// CIE xy chromaticity coordinates.
///
/// Used to specify custom white points and primaries.
/// Values are in the CIE 1931 xy chromaticity space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CIExy {
    /// CIE x coordinate.
    pub x: f64,
    /// CIE y coordinate.
    pub y: f64,
}

impl CIExy {
    /// Creates a new CIE xy coordinate pair.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Custom primaries specified as three CIE xy coordinate pairs (red, green, blue).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CustomPrimaries {
    /// Red primary CIE xy coordinates.
    pub red: CIExy,
    /// Green primary CIE xy coordinates.
    pub green: CIExy,
    /// Blue primary CIE xy coordinates.
    pub blue: CIExy,
}

/// Rough limit for CIE xy coordinate values (absolute value must be less than this).
const CUSTOMXY_ROUGH_LIMIT: f64 = 4.0;

/// Multiplier for converting CIE xy float values to fixed-point integers.
const CUSTOMXY_MUL: u32 = 1_000_000;

/// Minimum allowed fixed-point value for a custom xy coordinate.
const CUSTOMXY_MIN: i32 = -0x200000;

/// Maximum allowed fixed-point value for a custom xy coordinate.
const CUSTOMXY_MAX: i32 = 0x1FFFFF;

/// Encodes a signed integer using JXL's PackSigned encoding.
///
/// Maps non-negative X to 2*X, negative -X to 2*X-1.
/// This matches libjxl's `PackSigned` in `pack_signed.h`.
fn pack_signed(value: i32) -> u32 {
    ((value as u32) << 1) ^ (((!(value as u32)) >> 31).wrapping_sub(1))
}

/// Validates and converts a CIE xy float coordinate to a fixed-point integer.
///
/// Returns `Error::InvalidInput` if the value is out of range.
fn xy_to_fixed(value: f64, name: &str) -> Result<i32> {
    if value.abs() >= CUSTOMXY_ROUGH_LIMIT {
        return Err(Error::InvalidInput(format!(
            "custom {name} coordinate {value} out of range (must be < {CUSTOMXY_ROUGH_LIMIT})"
        )));
    }
    let fixed = (value * f64::from(CUSTOMXY_MUL)).round() as i32;
    if !(CUSTOMXY_MIN..=CUSTOMXY_MAX).contains(&fixed) {
        return Err(Error::InvalidInput(format!(
            "custom {name} coordinate {value} (fixed-point {fixed}) out of range [{CUSTOMXY_MIN}, {CUSTOMXY_MAX}]"
        )));
    }
    Ok(fixed)
}

/// Writes a single custom xy coordinate to the bitstream.
///
/// Uses the JXL U32 encoding with distribution:
/// - Selector 0: Bits(19), offset 0 — values 0..524287
/// - Selector 1: BitsOffset(19, 524288) — values 524288..1048575
/// - Selector 2: BitsOffset(20, 1048576) — values 1048576..2097151
/// - Selector 3: BitsOffset(21, 2097152) — values 2097152..4194303
///
/// The input is a signed fixed-point integer that is first PackSigned'd.
fn write_customxy_value(writer: &mut BitWriter, value: i32, name: &str) -> Result<()> {
    let _ = name; // Used only in trace macros (compiled out without trace-bitstream feature)
    let packed = pack_signed(value);

    if packed < 524288 {
        // Selector 0: Bits(19)
        crate::trace::debug_eprintln!(
            "CENC [bit {}]: {name} = {value} (packed {packed}, selector 0, 19 bits)",
            writer.bits_written()
        );
        writer.write(2, 0)?;
        writer.write(19, packed as u64)?;
    } else if packed < 1048576 {
        // Selector 1: BitsOffset(19, 524288)
        crate::trace::debug_eprintln!(
            "CENC [bit {}]: {name} = {value} (packed {packed}, selector 1, 19 bits + offset 524288)",
            writer.bits_written()
        );
        writer.write(2, 1)?;
        writer.write(19, (packed - 524288) as u64)?;
    } else if packed < 2097152 {
        // Selector 2: BitsOffset(20, 1048576)
        crate::trace::debug_eprintln!(
            "CENC [bit {}]: {name} = {value} (packed {packed}, selector 2, 20 bits + offset 1048576)",
            writer.bits_written()
        );
        writer.write(2, 2)?;
        writer.write(20, (packed - 1048576) as u64)?;
    } else {
        // Selector 3: BitsOffset(21, 2097152)
        crate::trace::debug_eprintln!(
            "CENC [bit {}]: {name} = {value} (packed {packed}, selector 3, 21 bits + offset 2097152)",
            writer.bits_written()
        );
        writer.write(2, 3)?;
        writer.write(21, (packed - 2097152) as u64)?;
    }
    Ok(())
}

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
    /// Custom white point CIE xy coordinates (required when `white_point == WhitePoint::Custom`).
    pub custom_white_point: Option<CIExy>,
    /// Primaries (for RGB).
    pub primaries: Primaries,
    /// Custom primaries (required when `primaries == Primaries::Custom`).
    pub custom_primaries: Option<CustomPrimaries>,
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
    /// Whether every serialized field equals the JXL **spec** default,
    /// so a writer may emit `ColorEncoding.all_default = 1` (one bit).
    ///
    /// Spec defaults from libjxl `d089091a`
    /// `color_encoding_internal.cc::ColorEncoding::VisitFields`:
    /// `want_icc = false`, `color_space = kRGB`, `white_point = kD65`,
    /// `primaries = kSRGB`, transfer function = `kSRGB` (i.e. *not* a
    /// custom gamma), `rendering_intent = kRelative`.
    ///
    /// Note the rendering-intent trap: [`Self::srgb`] builds
    /// `Perceptual`, which is **not** the spec default. The VarDCT
    /// header builder overwrites it with `Relative` to match libjxl, so
    /// the lossy path qualifies and a caller-supplied `srgb()` used
    /// verbatim would not. Testing the field rather than the
    /// constructor is deliberate.
    pub(crate) fn is_spec_default(&self) -> bool {
        !self.want_icc
            && self.color_space == ColorSpace::Rgb
            && self.white_point == WhitePoint::D65
            && self.custom_white_point.is_none()
            && self.primaries == Primaries::Srgb
            && self.custom_primaries.is_none()
            && self.gamma.is_none()
            && self.transfer_function == TransferFunction::Srgb
            && self.rendering_intent == RenderingIntent::Relative
    }

    /// Creates a standard sRGB color encoding.
    pub fn srgb() -> Self {
        Self {
            color_space: ColorSpace::Rgb,
            white_point: WhitePoint::D65,
            custom_white_point: None,
            primaries: Primaries::Srgb,
            custom_primaries: None,
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
            custom_white_point: None,
            primaries: Primaries::Srgb,
            custom_primaries: None,
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
            custom_white_point: None,
            primaries: Primaries::Srgb,
            custom_primaries: None,
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
            custom_white_point: None,
            primaries: Primaries::P3,
            custom_primaries: None,
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
            custom_white_point: None,
            primaries: Primaries::Bt2100,
            custom_primaries: None,
            transfer_function: TransferFunction::Pq,
            rendering_intent: RenderingIntent::Perceptual,
            want_icc: false,
            gamma: None,
        }
    }

    /// Creates a BT.2100 HLG (HDR) color encoding.
    pub fn bt2100_hlg() -> Self {
        Self {
            color_space: ColorSpace::Rgb,
            white_point: WhitePoint::D65,
            custom_white_point: None,
            primaries: Primaries::Bt2100,
            custom_primaries: None,
            transfer_function: TransferFunction::Hlg,
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
            custom_white_point: None,
            primaries: Primaries::Srgb,
            custom_primaries: None,
            transfer_function: TransferFunction::Srgb,
            rendering_intent: RenderingIntent::Perceptual,
            want_icc: false,
            gamma: None,
        }
    }

    /// Creates a color encoding with a custom white point.
    ///
    /// The CIE xy coordinates specify the white point in the CIE 1931 chromaticity diagram.
    /// For example, D50 is approximately (0.3457, 0.3585).
    pub fn with_custom_white_point(white_point: CIExy) -> Self {
        Self {
            white_point: WhitePoint::Custom,
            custom_white_point: Some(white_point),
            ..Self::srgb()
        }
    }

    /// Creates a color encoding with custom primaries.
    ///
    /// The three CIE xy coordinate pairs specify the red, green, and blue primaries.
    pub fn with_custom_primaries(primaries: CustomPrimaries) -> Self {
        Self {
            primaries: Primaries::Custom,
            custom_primaries: Some(primaries),
            ..Self::srgb()
        }
    }

    /// Creates a color encoding with both a custom white point and custom primaries.
    pub fn with_custom_white_point_and_primaries(
        white_point: CIExy,
        primaries: CustomPrimaries,
    ) -> Self {
        Self {
            white_point: WhitePoint::Custom,
            custom_white_point: Some(white_point),
            primaries: Primaries::Custom,
            custom_primaries: Some(primaries),
            ..Self::srgb()
        }
    }

    /// Builds a `ColorEncoding` from a CICP 4-tuple
    /// `(colour_primaries, transfer_characteristics, matrix_coefficients, full_range_flag)`.
    ///
    /// CICP (ITU-T H.273 / ISO/IEC 23091-2) is the wire-format used by Matroska,
    /// ISO BMFF, MP4, AV1, HEIC, and Ultra HDR. This helper maps the most common
    /// `(cp, tc, mc, full_range)` combinations to JXL's internal `ColorEncoding`.
    ///
    /// # Supported CICP values
    ///
    /// **Colour primaries** (`cp`):
    /// - `1` — BT.709 / sRGB (D65)
    /// - `9` — BT.2100 (D65, also used for BT.2020)
    /// - `11` — DCI-P3 (DCI white point, P3 primaries — SMPTE RP 431-2 movie projector)
    /// - `12` — Display P3 (D65 white point, P3 primaries — Apple's display gamut)
    ///
    /// **Transfer characteristics** (`tc`):
    /// - `1` — BT.709
    /// - `8` — Linear
    /// - `13` — sRGB (IEC 61966-2-1)
    /// - `16` — PQ (SMPTE ST 2084)
    /// - `17` — DCI gamma 2.6 (SMPTE ST 428-1)
    /// - `18` — HLG (BT.2100)
    ///
    /// **Matrix coefficients** (`mc`): must be `0` (Identity / RGB). JXL does not
    /// accept YCbCr at the encoder boundary; the caller must RGB-convert first.
    ///
    /// **Full range** (`full_range`): must be `true`. JXL's encoder works on
    /// full-range values; the caller is responsible for narrow→full conversion.
    ///
    /// Mapping matches libjxl's `ApplyCICP` in `lib/jxl/cms/jxl_cms.cc:928`.
    ///
    /// # Errors
    ///
    /// Returns `Err(&'static str)` with a descriptive message when:
    /// - `mc != 0` (non-RGB matrix coefficients — convert to RGB first)
    /// - `!full_range` (limited / narrow range — convert to full range first)
    /// - `cp` is not one of `{1, 9, 11, 12}`
    /// - `tc` is not one of `{1, 8, 13, 16, 17, 18}`
    ///
    /// # Examples
    ///
    /// ```
    /// use jxl_encoder::ColorEncoding;
    /// // sRGB
    /// let srgb = ColorEncoding::from_cicp(1, 13, 0, true).unwrap();
    /// assert!(srgb.is_srgb());
    ///
    /// // BT.2100 PQ HDR
    /// let pq = ColorEncoding::from_cicp(9, 16, 0, true).unwrap();
    /// assert_eq!(pq.transfer_function as u8, 16);
    /// ```
    pub fn from_cicp(
        cp: u8,
        tc: u8,
        mc: u8,
        full_range: bool,
    ) -> core::result::Result<Self, &'static str> {
        // libjxl ApplyCICP: matrix_coefficients must be 0 (Identity / RGB).
        if mc != 0 {
            return Err("CICP matrix_coefficients != 0 (YCbCr) is not supported; \
                 convert pixels to RGB before encoding");
        }
        // libjxl ApplyCICP: full_range must be 1.
        if !full_range {
            return Err(
                "CICP video_full_range_flag == 0 (limited range) is not supported; \
                 convert pixels to full range before encoding",
            );
        }

        // Transfer characteristics → TransferFunction (matches libjxl enum values).
        let transfer_function = match tc {
            1 => TransferFunction::Bt709,
            8 => TransferFunction::Linear,
            13 => TransferFunction::Srgb,
            16 => TransferFunction::Pq,
            17 => TransferFunction::Dci,
            18 => TransferFunction::Hlg,
            _ => return Err("CICP transfer_characteristics not in {1, 8, 13, 16, 17, 18}"),
        };

        // Colour primaries → (Primaries, WhitePoint).
        //
        // Per libjxl ApplyCICP:
        // - cp == 11 (Primaries::kP3 in libjxl's enum) → WhitePoint::DCI + Primaries::P3
        //   (i.e., DCI-P3 movie projector)
        // - cp == 12 (kColorPrimariesP3_D65, not a libjxl Primaries enum value) →
        //   WhitePoint::D65 + Primaries::P3 (i.e., Apple Display P3)
        // - cp == 1 → WhitePoint::D65 + Primaries::Srgb
        // - cp == 9 → WhitePoint::D65 + Primaries::Bt2100
        let (white_point, primaries) = match cp {
            1 => (WhitePoint::D65, Primaries::Srgb),
            9 => (WhitePoint::D65, Primaries::Bt2100),
            11 => (WhitePoint::Dci, Primaries::P3),
            12 => (WhitePoint::D65, Primaries::P3),
            _ => return Err("CICP colour_primaries not in {1, 9, 11, 12}"),
        };

        Ok(Self {
            color_space: ColorSpace::Rgb,
            white_point,
            custom_white_point: None,
            primaries,
            custom_primaries: None,
            transfer_function,
            rendering_intent: RenderingIntent::Perceptual,
            want_icc: false,
            gamma: None,
        })
    }

    /// Returns true if this matches the JXL default color encoding.
    /// (sRGB with Perceptual rendering intent, no ICC)
    ///
    /// When all_default=true for metadata with xyb_encoded=true (lossy mode),
    /// the decoder assumes sRGB input color space.
    pub fn is_srgb(&self) -> bool {
        self.color_space == ColorSpace::Rgb
            && self.white_point == WhitePoint::D65
            && self.custom_white_point.is_none()
            && self.primaries == Primaries::Srgb
            && self.custom_primaries.is_none()
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
            let wp_xy = self.custom_white_point.ok_or_else(|| {
                Error::InvalidInput(
                    "custom_white_point must be set when white_point is Custom".into(),
                )
            })?;
            let wx = xy_to_fixed(wp_xy.x, "white_point.x")?;
            let wy = xy_to_fixed(wp_xy.y, "white_point.y")?;
            write_customxy_value(writer, wx, "white_point.x")?;
            write_customxy_value(writer, wy, "white_point.y")?;
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
                let cp = self.custom_primaries.ok_or_else(|| {
                    Error::InvalidInput(
                        "custom_primaries must be set when primaries is Custom".into(),
                    )
                })?;
                // Red primary
                let rx = xy_to_fixed(cp.red.x, "red.x")?;
                let ry = xy_to_fixed(cp.red.y, "red.y")?;
                write_customxy_value(writer, rx, "red.x")?;
                write_customxy_value(writer, ry, "red.y")?;
                // Green primary
                let gx = xy_to_fixed(cp.green.x, "green.x")?;
                let gy = xy_to_fixed(cp.green.y, "green.y")?;
                write_customxy_value(writer, gx, "green.x")?;
                write_customxy_value(writer, gy, "green.y")?;
                // Blue primary
                let bx = xy_to_fixed(cp.blue.x, "blue.x")?;
                let by = xy_to_fixed(cp.blue.y, "blue.y")?;
                write_customxy_value(writer, bx, "blue.x")?;
                write_customxy_value(writer, by, "blue.y")?;
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
            rendering_intent: RenderingIntent::Relative, // Non-default
            ..ColorEncoding::srgb()
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
    fn test_write_bt2100_hlg() {
        let enc = ColorEncoding::bt2100_hlg();
        assert_eq!(enc.color_space, ColorSpace::Rgb);
        assert_eq!(enc.white_point, WhitePoint::D65);
        assert_eq!(enc.primaries, Primaries::Bt2100);
        assert_eq!(enc.transfer_function, TransferFunction::Hlg);
        assert_eq!(enc.rendering_intent, RenderingIntent::Perceptual);
        assert!(!enc.want_icc);

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

    // ---- pack_signed tests ----

    #[test]
    fn test_pack_signed_zero() {
        assert_eq!(pack_signed(0), 0);
    }

    #[test]
    fn test_pack_signed_positive() {
        assert_eq!(pack_signed(1), 2);
        assert_eq!(pack_signed(2), 4);
        assert_eq!(pack_signed(100), 200);
    }

    #[test]
    fn test_pack_signed_negative() {
        assert_eq!(pack_signed(-1), 1);
        assert_eq!(pack_signed(-2), 3);
        assert_eq!(pack_signed(-100), 199);
    }

    #[test]
    fn test_pack_signed_roundtrip() {
        // Verify PackSigned is invertible (matches libjxl UnpackSigned)
        for v in [-1000000, -1, 0, 1, 1000000, CUSTOMXY_MIN, CUSTOMXY_MAX] {
            let packed = pack_signed(v);
            // UnpackSigned: (packed >> 1) ^ (((!(packed)) & 1).wrapping_sub(1))
            let unpacked = (packed >> 1) as i32 ^ (((!packed) & 1).wrapping_sub(1)) as i32;
            assert_eq!(unpacked, v, "pack_signed roundtrip failed for {v}");
        }
    }

    // ---- xy_to_fixed tests ----

    #[test]
    fn test_xy_to_fixed_d65() {
        // D65 white point: (0.3127, 0.3290)
        let x = xy_to_fixed(0.3127, "x").unwrap();
        let y = xy_to_fixed(0.3290, "y").unwrap();
        assert_eq!(x, 312700);
        assert_eq!(y, 329000);
    }

    #[test]
    fn test_xy_to_fixed_out_of_range() {
        // Values >= 4.0 should fail
        assert!(xy_to_fixed(4.0, "x").is_err());
        assert!(xy_to_fixed(-4.0, "x").is_err());
        // Values within rough limit but outside fixed-point range
        assert!(xy_to_fixed(3.9, "x").is_err());
    }

    #[test]
    fn test_xy_to_fixed_negative() {
        let v = xy_to_fixed(-0.5, "x").unwrap();
        assert_eq!(v, -500000);
    }

    // ---- Custom white point encoding tests ----

    #[test]
    fn test_write_custom_white_point_d50() {
        // D50 white point: (0.3457, 0.3585)
        let enc = ColorEncoding::with_custom_white_point(CIExy::new(0.3457, 0.3585));
        assert_eq!(enc.white_point, WhitePoint::Custom);
        assert!(enc.custom_white_point.is_some());
        assert!(!enc.is_srgb());

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        // Should produce a valid bitstream with custom white point data
        assert!(writer.bits_written() > 0);
        // Custom white point adds: 2 coordinates * (2 selector + 19-21 data) bits each
        // This should be longer than the standard D65 encoding
    }

    #[test]
    fn test_write_custom_white_point_missing_coordinates() {
        // Set white_point to Custom but don't provide coordinates
        let enc = ColorEncoding {
            white_point: WhitePoint::Custom,
            custom_white_point: None,
            ..ColorEncoding::srgb()
        };

        let mut writer = BitWriter::new();
        let result = enc.write(&mut writer);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("custom_white_point must be set"),
            "unexpected error: {err}"
        );
    }

    // ---- Custom primaries encoding tests ----

    #[test]
    fn test_write_custom_primaries() {
        // Adobe RGB primaries
        let primaries = CustomPrimaries {
            red: CIExy::new(0.6400, 0.3300),
            green: CIExy::new(0.2100, 0.7100),
            blue: CIExy::new(0.1500, 0.0600),
        };
        let enc = ColorEncoding::with_custom_primaries(primaries);
        assert_eq!(enc.primaries, Primaries::Custom);
        assert!(enc.custom_primaries.is_some());
        assert!(!enc.is_srgb());

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        // Should produce a valid bitstream with 6 custom xy values
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_custom_primaries_missing_coordinates() {
        let enc = ColorEncoding {
            primaries: Primaries::Custom,
            custom_primaries: None,
            ..ColorEncoding::srgb()
        };

        let mut writer = BitWriter::new();
        let result = enc.write(&mut writer);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("custom_primaries must be set"),
            "unexpected error: {err}"
        );
    }

    // ---- Combined custom white point + primaries ----

    #[test]
    fn test_write_custom_white_point_and_primaries() {
        let enc = ColorEncoding::with_custom_white_point_and_primaries(
            CIExy::new(0.3457, 0.3585), // D50
            CustomPrimaries {
                red: CIExy::new(0.7347, 0.2653),
                green: CIExy::new(0.1596, 0.8404),
                blue: CIExy::new(0.0366, 0.0001),
            },
        );
        assert_eq!(enc.white_point, WhitePoint::Custom);
        assert_eq!(enc.primaries, Primaries::Custom);

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        // 2 wp coordinates + 6 primary coordinates = 8 customxy values
        assert!(writer.bits_written() > 0);
    }

    // ---- Bit-level encoding verification ----

    #[test]
    fn test_customxy_encoding_small_positive() {
        // A small positive value should use selector 0 (Bits(19))
        // D65 x = 0.3127 → fixed 312700 → packed = 625400
        // 625400 > 524287 → selector 1 (BitsOffset(19, 524288))
        let mut writer = BitWriter::new();
        write_customxy_value(&mut writer, 312700, "test").unwrap();
        let packed = pack_signed(312700);
        assert_eq!(packed, 625400);
        // selector 1 → 2 + 19 = 21 bits
        assert_eq!(writer.bits_written(), 21);
    }

    #[test]
    fn test_customxy_encoding_zero() {
        // Zero should use selector 0
        let mut writer = BitWriter::new();
        write_customxy_value(&mut writer, 0, "test").unwrap();
        assert_eq!(pack_signed(0), 0);
        // selector 0 → 2 + 19 = 21 bits
        assert_eq!(writer.bits_written(), 21);
    }

    #[test]
    fn test_customxy_encoding_negative() {
        // -1 → packed 1, selector 0
        let mut writer = BitWriter::new();
        write_customxy_value(&mut writer, -1, "test").unwrap();
        assert_eq!(pack_signed(-1), 1);
        assert_eq!(writer.bits_written(), 21); // selector 0: 2 + 19
    }

    #[test]
    fn test_customxy_encoding_all_selectors() {
        // Verify each selector is chosen correctly based on packed value ranges

        // Selector 0: packed 0..524287 (Bits(19))
        let mut w = BitWriter::new();
        // value 262143 → packed 524286
        write_customxy_value(&mut w, 262143, "test").unwrap();
        assert_eq!(w.bits_written(), 21); // 2 + 19

        // Selector 1: packed 524288..1048575 (BitsOffset(19, 524288))
        let mut w = BitWriter::new();
        // value 262144 → packed 524288
        write_customxy_value(&mut w, 262144, "test").unwrap();
        assert_eq!(w.bits_written(), 21); // 2 + 19

        // Selector 2: packed 1048576..2097151 (BitsOffset(20, 1048576))
        let mut w = BitWriter::new();
        // value 524288 → packed 1048576
        write_customxy_value(&mut w, 524288, "test").unwrap();
        assert_eq!(w.bits_written(), 22); // 2 + 20

        // Selector 3: packed 2097152..4194303 (BitsOffset(21, 2097152))
        let mut w = BitWriter::new();
        // value 1048576 → packed 2097152
        write_customxy_value(&mut w, 1048576, "test").unwrap();
        assert_eq!(w.bits_written(), 23); // 2 + 21
    }

    #[test]
    fn test_write_custom_wp_bit_count_vs_standard() {
        // Custom white point encoding should use more bits than D65
        let enc_d65 = ColorEncoding {
            rendering_intent: RenderingIntent::Relative,
            ..ColorEncoding::srgb()
        };
        let enc_custom = ColorEncoding {
            white_point: WhitePoint::Custom,
            custom_white_point: Some(CIExy::new(0.3127, 0.3290)),
            rendering_intent: RenderingIntent::Relative,
            ..ColorEncoding::srgb()
        };

        let mut w_d65 = BitWriter::new();
        enc_d65.write(&mut w_d65).unwrap();
        let bits_d65 = w_d65.bits_written();

        let mut w_custom = BitWriter::new();
        enc_custom.write(&mut w_custom).unwrap();
        let bits_custom = w_custom.bits_written();

        assert!(
            bits_custom > bits_d65,
            "custom WP should use more bits: {bits_custom} vs {bits_d65}"
        );
    }

    #[test]
    fn test_default_encoding_custom_fields() {
        let enc = ColorEncoding::default();
        assert!(enc.custom_white_point.is_none());
        assert!(enc.custom_primaries.is_none());
    }

    // ---- Grayscale with custom white point (no primaries written) ----

    #[test]
    fn test_write_grayscale_custom_white_point() {
        let enc = ColorEncoding {
            color_space: ColorSpace::Gray,
            white_point: WhitePoint::Custom,
            custom_white_point: Some(CIExy::new(0.3457, 0.3585)),
            ..ColorEncoding::gray()
        };

        let mut writer = BitWriter::new();
        enc.write(&mut writer).unwrap();
        // Grayscale skips primaries entirely, but custom WP should still be written
        assert!(writer.bits_written() > 0);
    }

    // ---- Roundtrip decode tests with jxl-rs ----

    #[test]
    fn test_roundtrip_custom_white_point_d50() {
        // Encode a small image with D50 custom white point, decode with jxl-rs
        let width = 16u32;
        let height = 16u32;
        let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

        let ce = ColorEncoding::with_custom_white_point(CIExy::new(0.3457, 0.3585));

        let encoded = crate::LosslessConfig::new()
            .encode_request(width, height, crate::PixelLayout::Rgb8)
            .with_color_encoding(ce)
            .encode(&pixels)
            .expect("encoding with custom white point should succeed");

        // Decode with jxl-rs (primary decoder)
        let decoded = crate::test_helpers::decode_with_jxl_rs(&encoded)
            .expect("jxl-rs should decode custom white point");
        assert_eq!(decoded.width, width as usize);
        assert_eq!(decoded.height, height as usize);
    }

    #[test]
    fn test_roundtrip_custom_primaries() {
        // Encode with Adobe RGB-like custom primaries
        let width = 16u32;
        let height = 16u32;
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 7) % 256) as u8)
            .collect();

        let ce = ColorEncoding::with_custom_primaries(CustomPrimaries {
            red: CIExy::new(0.6400, 0.3300),
            green: CIExy::new(0.2100, 0.7100),
            blue: CIExy::new(0.1500, 0.0600),
        });

        let encoded = crate::LosslessConfig::new()
            .encode_request(width, height, crate::PixelLayout::Rgb8)
            .with_color_encoding(ce)
            .encode(&pixels)
            .expect("encoding with custom primaries should succeed");

        let decoded = crate::test_helpers::decode_with_jxl_rs(&encoded)
            .expect("jxl-rs should decode custom primaries");
        assert_eq!(decoded.width, width as usize);
        assert_eq!(decoded.height, height as usize);
    }

    #[test]
    fn test_roundtrip_custom_white_point_and_primaries() {
        // ProPhoto RGB: D50 white point + wide gamut primaries
        let width = 16u32;
        let height = 16u32;
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 13) % 256) as u8)
            .collect();

        let ce = ColorEncoding::with_custom_white_point_and_primaries(
            CIExy::new(0.3457, 0.3585), // D50
            CustomPrimaries {
                red: CIExy::new(0.7347, 0.2653),
                green: CIExy::new(0.1596, 0.8404),
                blue: CIExy::new(0.0366, 0.0001),
            },
        );

        let encoded = crate::LosslessConfig::new()
            .encode_request(width, height, crate::PixelLayout::Rgb8)
            .with_color_encoding(ce)
            .encode(&pixels)
            .expect("encoding with custom WP + primaries should succeed");

        let decoded = crate::test_helpers::decode_with_jxl_rs(&encoded)
            .expect("jxl-rs should decode custom WP + primaries");
        assert_eq!(decoded.width, width as usize);
        assert_eq!(decoded.height, height as usize);
    }

    // ---- CICP `from_cicp` lookup tests ----

    #[test]
    fn test_from_cicp_srgb() {
        // CICP (1, 13, 0, true) = sRGB
        let ce = ColorEncoding::from_cicp(1, 13, 0, true).expect("sRGB CICP should parse");
        assert_eq!(ce.color_space, ColorSpace::Rgb);
        assert_eq!(ce.white_point, WhitePoint::D65);
        assert_eq!(ce.primaries, Primaries::Srgb);
        assert_eq!(ce.transfer_function, TransferFunction::Srgb);
        assert_eq!(ce.rendering_intent, RenderingIntent::Perceptual);
        assert!(!ce.want_icc);
        assert!(ce.is_srgb(), "CICP (1,13,0,1) must be exactly sRGB default");
    }

    #[test]
    fn test_from_cicp_bt709() {
        // CICP (1, 1, 0, true) = BT.709 primaries + BT.709 transfer
        let ce = ColorEncoding::from_cicp(1, 1, 0, true).expect("BT.709 CICP should parse");
        assert_eq!(ce.color_space, ColorSpace::Rgb);
        assert_eq!(ce.white_point, WhitePoint::D65);
        assert_eq!(ce.primaries, Primaries::Srgb); // BT.709 shares sRGB primaries
        assert_eq!(ce.transfer_function, TransferFunction::Bt709);
        assert!(!ce.is_srgb()); // Different TF from sRGB
    }

    #[test]
    fn test_from_cicp_bt2100_pq() {
        // CICP (9, 16, 0, true) = Rec.2100 PQ
        let ce = ColorEncoding::from_cicp(9, 16, 0, true).expect("BT.2100 PQ should parse");
        assert_eq!(ce.color_space, ColorSpace::Rgb);
        assert_eq!(ce.white_point, WhitePoint::D65);
        assert_eq!(ce.primaries, Primaries::Bt2100);
        assert_eq!(ce.transfer_function, TransferFunction::Pq);
        // Equivalent to the existing preset constructor
        let preset = ColorEncoding::bt2100_pq();
        assert_eq!(ce.color_space, preset.color_space);
        assert_eq!(ce.white_point, preset.white_point);
        assert_eq!(ce.primaries, preset.primaries);
        assert_eq!(ce.transfer_function, preset.transfer_function);
    }

    #[test]
    fn test_from_cicp_bt2100_hlg() {
        // CICP (9, 18, 0, true) = Rec.2100 HLG
        let ce = ColorEncoding::from_cicp(9, 18, 0, true).expect("BT.2100 HLG should parse");
        assert_eq!(ce.color_space, ColorSpace::Rgb);
        assert_eq!(ce.white_point, WhitePoint::D65);
        assert_eq!(ce.primaries, Primaries::Bt2100);
        assert_eq!(ce.transfer_function, TransferFunction::Hlg);
        let preset = ColorEncoding::bt2100_hlg();
        assert_eq!(ce.color_space, preset.color_space);
        assert_eq!(ce.white_point, preset.white_point);
        assert_eq!(ce.primaries, preset.primaries);
        assert_eq!(ce.transfer_function, preset.transfer_function);
    }

    #[test]
    fn test_from_cicp_dci_p3_with_dci_tf() {
        // CICP (12, 17, 0, true) — Apple Display P3 primaries (cp=12 = P3+D65),
        // with DCI gamma 2.6 transfer (tc=17). Matches libjxl ApplyCICP exactly:
        // cp=12 → kColorPrimariesP3_D65 → (WhitePoint::D65, Primaries::P3).
        let ce = ColorEncoding::from_cicp(12, 17, 0, true).expect("CP=12, TC=17 should parse");
        assert_eq!(ce.color_space, ColorSpace::Rgb);
        assert_eq!(ce.white_point, WhitePoint::D65);
        assert_eq!(ce.primaries, Primaries::P3);
        assert_eq!(ce.transfer_function, TransferFunction::Dci);
    }

    #[test]
    fn test_from_cicp_dci_p3_movie_projector() {
        // CICP (11, 17, 0, true) — true DCI-P3 movie projector:
        // cp=11 → (WhitePoint::DCI, Primaries::P3), tc=17 (DCI gamma 2.6).
        let ce = ColorEncoding::from_cicp(11, 17, 0, true).expect("DCI-P3 should parse");
        assert_eq!(ce.color_space, ColorSpace::Rgb);
        assert_eq!(ce.white_point, WhitePoint::Dci);
        assert_eq!(ce.primaries, Primaries::P3);
        assert_eq!(ce.transfer_function, TransferFunction::Dci);
    }

    #[test]
    fn test_from_cicp_display_p3_with_srgb_tf() {
        // CICP (12, 13, 0, true) = Display P3 (Apple): P3 primaries + D65 WP + sRGB TF.
        // Equivalent to `display_p3()` preset.
        let ce = ColorEncoding::from_cicp(12, 13, 0, true).expect("Display P3 should parse");
        let preset = ColorEncoding::display_p3();
        assert_eq!(ce.color_space, preset.color_space);
        assert_eq!(ce.white_point, preset.white_point);
        assert_eq!(ce.primaries, preset.primaries);
        assert_eq!(ce.transfer_function, preset.transfer_function);
    }

    #[test]
    fn test_from_cicp_linear_srgb() {
        // CICP (1, 8, 0, true) = sRGB primaries with Linear transfer.
        let ce = ColorEncoding::from_cicp(1, 8, 0, true).expect("Linear sRGB should parse");
        assert_eq!(ce.transfer_function, TransferFunction::Linear);
        let preset = ColorEncoding::linear_srgb();
        assert_eq!(ce.color_space, preset.color_space);
        assert_eq!(ce.white_point, preset.white_point);
        assert_eq!(ce.primaries, preset.primaries);
        assert_eq!(ce.transfer_function, preset.transfer_function);
    }

    #[test]
    fn test_from_cicp_rejects_ycbcr() {
        // matrix_coefficients != 0 means YCbCr (or some other non-RGB matrix).
        // JXL does not accept YCbCr at the encoder boundary.
        let err =
            ColorEncoding::from_cicp(1, 13, 1, true).expect_err("mc=1 (BT.709 YCbCr) must reject");
        assert!(
            err.contains("matrix_coefficients"),
            "error should mention matrix_coefficients, got: {err}"
        );
    }

    #[test]
    fn test_from_cicp_rejects_limited_range() {
        // full_range=false means narrow-range / video-range pixels.
        let err =
            ColorEncoding::from_cicp(1, 13, 0, false).expect_err("full_range=false must reject");
        assert!(
            err.contains("full_range") || err.contains("limited range"),
            "error should mention full_range, got: {err}"
        );
    }

    #[test]
    fn test_from_cicp_rejects_unknown_primaries() {
        // cp=2 is "unspecified" in CICP, cp=4 is BT.470M (not supported by JXL).
        for unsupported_cp in [0u8, 2, 3, 4, 5, 6, 7, 8, 10, 13, 22, 255] {
            let err = ColorEncoding::from_cicp(unsupported_cp, 13, 0, true)
                .expect_err("unsupported colour_primaries must reject");
            assert!(
                err.contains("colour_primaries"),
                "cp={unsupported_cp}: error should mention colour_primaries, got: {err}"
            );
        }
    }

    #[test]
    fn test_from_cicp_rejects_unknown_transfer() {
        // tc=2 is "unspecified", tc=4 is BT.470M gamma 2.2 (we use the gamma path).
        for unsupported_tc in [0u8, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 19, 255] {
            let err = ColorEncoding::from_cicp(1, unsupported_tc, 0, true)
                .expect_err("unsupported transfer_characteristics must reject");
            assert!(
                err.contains("transfer_characteristics"),
                "tc={unsupported_tc}: error should mention transfer_characteristics, got: {err}"
            );
        }
    }

    #[test]
    fn test_from_cicp_roundtrips_through_bitstream() {
        // Every supported CICP combo must serialize to a valid color_encoding bitstream.
        // This is the layer-2 invariant: from_cicp() output must encode without error.
        let cases: &[(u8, u8, &str)] = &[
            (1, 13, "sRGB"),
            (1, 1, "BT.709"),
            (1, 8, "Linear sRGB"),
            (9, 16, "BT.2100 PQ"),
            (9, 18, "BT.2100 HLG"),
            (9, 13, "BT.2100 + sRGB TF (unusual but valid)"),
            (9, 8, "BT.2100 + Linear TF"),
            (11, 17, "DCI-P3 movie projector"),
            (11, 13, "DCI-P3 primaries + sRGB TF"),
            (12, 13, "Display P3 (Apple)"),
            (12, 16, "Display P3 + PQ (HDR with P3 gamut)"),
            (12, 17, "Display P3 primaries + DCI TF"),
        ];
        for (cp, tc, name) in cases {
            let ce = ColorEncoding::from_cicp(*cp, *tc, 0, true)
                .unwrap_or_else(|e| panic!("CICP ({cp},{tc},0,1) [{name}] failed: {e}"));
            let mut writer = BitWriter::new();
            ce.write(&mut writer)
                .unwrap_or_else(|e| panic!("CICP ({cp},{tc},0,1) [{name}] bitstream failed: {e}"));
            assert!(
                writer.bits_written() > 0,
                "CICP ({cp},{tc},0,1) [{name}] wrote zero bits"
            );
        }
    }

    #[test]
    fn test_from_cicp_roundtrip_srgb_with_jxl_rs() {
        // Encode a small image with CICP-derived sRGB encoding, decode with jxl-rs.
        let width = 16u32;
        let height = 16u32;
        let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

        let ce = ColorEncoding::from_cicp(1, 13, 0, true).unwrap();
        let encoded = crate::LosslessConfig::new()
            .encode_request(width, height, crate::PixelLayout::Rgb8)
            .with_color_encoding(ce)
            .encode(&pixels)
            .expect("CICP-derived sRGB encoding should succeed");

        let decoded = crate::test_helpers::decode_with_jxl_rs(&encoded)
            .expect("jxl-rs should decode CICP-derived sRGB");
        assert_eq!(decoded.width, width as usize);
        assert_eq!(decoded.height, height as usize);
    }

    #[test]
    fn test_from_cicp_roundtrip_bt2100_pq_with_jxl_rs() {
        // Encode a small image with BT.2100 PQ via CICP, decode with jxl-rs.
        let width = 16u32;
        let height = 16u32;
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 7) % 256) as u8)
            .collect();

        let ce = ColorEncoding::from_cicp(9, 16, 0, true).unwrap();
        let encoded = crate::LosslessConfig::new()
            .encode_request(width, height, crate::PixelLayout::Rgb8)
            .with_color_encoding(ce)
            .encode(&pixels)
            .expect("CICP-derived BT.2100 PQ should succeed");

        let decoded = crate::test_helpers::decode_with_jxl_rs(&encoded)
            .expect("jxl-rs should decode CICP-derived BT.2100 PQ");
        assert_eq!(decoded.width, width as usize);
        assert_eq!(decoded.height, height as usize);
    }
}
