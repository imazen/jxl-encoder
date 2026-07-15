// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Input pixel format descriptor (\`PixelLayout\`).

/// Describes the pixel format of input data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PixelLayout {
    /// 8-bit sRGB, 3 bytes per pixel (R, G, B).
    Rgb8,
    /// 8-bit sRGB + alpha, 4 bytes per pixel (R, G, B, A).
    Rgba8,
    /// 8-bit sRGB in BGR order, 3 bytes per pixel (B, G, R).
    Bgr8,
    /// 8-bit sRGB in BGRA order, 4 bytes per pixel (B, G, R, A).
    Bgra8,
    /// 8-bit grayscale, 1 byte per pixel.
    Gray8,
    /// 8-bit grayscale + alpha, 2 bytes per pixel.
    GrayAlpha8,
    /// 16-bit sRGB, 6 bytes per pixel (R, G, B) — native-endian u16.
    Rgb16,
    /// 16-bit sRGB + alpha, 8 bytes per pixel (R, G, B, A) — native-endian u16.
    Rgba16,
    /// 16-bit grayscale, 2 bytes per pixel — native-endian u16.
    Gray16,
    /// 16-bit grayscale + alpha, 4 bytes per pixel — native-endian u16.
    GrayAlpha16,
    /// Linear f32 RGB, 12 bytes per pixel. Skips sRGB→linear conversion.
    RgbLinearF32,
    /// Linear f32 RGBA, 16 bytes per pixel. Skips sRGB→linear conversion.
    RgbaLinearF32,
    /// Linear f32 grayscale, 4 bytes per pixel.
    GrayLinearF32,
    /// Linear f32 grayscale + alpha, 8 bytes per pixel.
    GrayAlphaLinearF32,
    /// Linear IEEE 754 binary16 RGB, 6 bytes per pixel — native-endian
    /// u16. Common for GPU pipelines (WebGPU, CUDA, Metal, Vulkan,
    /// Direct2D). Same semantics as [`Self::RgbLinearF32`] but at
    /// half precision; encoder converts to f32 internally before XYB.
    RgbLinearF16,
    /// Linear IEEE 754 binary16 RGBA, 8 bytes per pixel — native-endian
    /// u16. See [`Self::RgbLinearF16`].
    RgbaLinearF16,
    /// Linear IEEE 754 binary16 grayscale, 2 bytes per pixel.
    GrayLinearF16,
    /// Linear IEEE 754 binary16 grayscale + alpha, 4 bytes per pixel.
    GrayAlphaLinearF16,
    /// PQ-encoded (SMPTE ST 2084) f32 RGB, 12 bytes per pixel. Same
    /// storage shape as [`Self::RgbLinearF32`] but the f32 values are
    /// interpreted as PQ-encoded `[0, 1]` and run through the inverse
    /// PQ EOTF before XYB. Use for HDR GPU/Vulkan/Metal pipelines that
    /// emit PQ floats directly. Closes A3 chunk 1b (issue #46).
    RgbPqF32,
    /// PQ-encoded f32 RGBA, 16 bytes per pixel. See [`Self::RgbPqF32`].
    RgbaPqF32,
    /// HLG-encoded (BT.2100 / ARIB STD-B67) f32 RGB, 12 bytes per
    /// pixel. f32 values are interpreted as HLG-encoded `[0, 1]` and
    /// run through the inverse HLG OETF (scene-light) before XYB.
    /// Closes A3 chunk 1b (issue #46).
    RgbHlgF32,
    /// HLG-encoded f32 RGBA, 16 bytes per pixel. See [`Self::RgbHlgF32`].
    RgbaHlgF32,
    /// BT.709 (Rec. ITU-R BT.709-6) gamma-encoded f32 RGB, 12 bytes
    /// per pixel. f32 values are interpreted as BT.709-encoded `[0, 1]`
    /// and run through the inverse BT.709 OETF before XYB. Closes A3
    /// chunk 1b (issue #46).
    RgbBt709F32,
    /// BT.709-encoded f32 RGBA, 16 bytes per pixel. See [`Self::RgbBt709F32`].
    RgbaBt709F32,
    /// 8-bit CMYK, 4 bytes per pixel (C, M, Y, K). The C/M/Y planes
    /// are encoded as the frame's 3 color channels; the K (Black)
    /// plane is encoded as an [`crate::headers::extra_channels::ExtraChannelType::Black`] extra
    /// channel. JXL convention is **`0 = full ink, 255 = no ink`**
    /// for all four planes (libjxl `enc_image_bundle.cc:65`); callers
    /// must pre-invert any "0 = no ink" CMYK input before encoding.
    ///
    /// For a colour-managed CMYK workflow attach a CMYK ICC profile
    /// via [`crate::EncodeRequest::with_metadata`] →
    /// [`crate::ImageMetadata::icc_profile`]; without an ICC the decoder will
    /// fall back to interpreting the CMY planes as sRGB and the K
    /// plane as an opaque extra channel. Both lossless and one-shot
    /// lossy encoding are wired: lossy routes C/M/Y through VarDCT
    /// (XYB) via a naive `1-CMY × (1-K)` subtractive transform
    /// (gamut-direction correct, not colorimetric — no ICC/SWOP
    /// calibration yet) while K rides the Black extra channel
    /// losslessly, so CMYK semantics survive the round-trip.
    /// Streaming CMYK is not yet wired (one-shot only). Closes #58.
    ///
    /// Bumps codestream level to 10 (level 5 forbids the Black
    /// extra channel; see `compute_codestream_level`).
    Cmyk8,
    /// 16-bit CMYK, 8 bytes per pixel — native-endian u16 per channel
    /// (C, M, Y, K). Same `0 = full ink, 65535 = no ink` convention
    /// as [`Self::Cmyk8`]; same one-shot lossless-and-lossy support. The
    /// Black channel is signaled as 16-bit in the extra-channel header.
    Cmyk16,
}

impl PixelLayout {
    /// Bytes per pixel for this layout.
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 | Self::Bgr8 => 3,
            Self::Rgba8 | Self::Bgra8 => 4,
            Self::Gray8 => 1,
            Self::GrayAlpha8 => 2,
            Self::Rgb16 => 6,
            Self::Rgba16 => 8,
            Self::Gray16 => 2,
            Self::GrayAlpha16 => 4,
            Self::RgbLinearF32 => 12,
            Self::RgbaLinearF32 => 16,
            Self::GrayLinearF32 => 4,
            Self::GrayAlphaLinearF32 => 8,
            Self::RgbLinearF16 => 6,
            Self::RgbaLinearF16 => 8,
            Self::GrayLinearF16 => 2,
            Self::GrayAlphaLinearF16 => 4,
            // A3 chunk 1b: f32 HDR/SDR-encoded variants (issue #46).
            // Same byte width as RgbLinearF32 / RgbaLinearF32; the
            // transfer function lives in the layout name, not the
            // storage size.
            Self::RgbPqF32 | Self::RgbHlgF32 | Self::RgbBt709F32 => 12,
            Self::RgbaPqF32 | Self::RgbaHlgF32 | Self::RgbaBt709F32 => 16,
            // CMYK is C, M, Y, K interleaved — 4 bytes (8-bit) or
            // 8 bytes (16-bit). The K plane is split off and encoded
            // as a Black extra channel; the on-disk codestream still
            // carries 3 colour channels.
            Self::Cmyk8 => 4,
            Self::Cmyk16 => 8,
        }
    }

    /// Whether this layout uses linear (not gamma-encoded) values.
    pub const fn is_linear(self) -> bool {
        matches!(
            self,
            Self::RgbLinearF32
                | Self::RgbaLinearF32
                | Self::GrayLinearF32
                | Self::GrayAlphaLinearF32
                | Self::RgbLinearF16
                | Self::RgbaLinearF16
                | Self::GrayLinearF16
                | Self::GrayAlphaLinearF16
        )
    }

    /// Whether this layout uses 16-bit samples.
    pub const fn is_16bit(self) -> bool {
        matches!(
            self,
            Self::Rgb16 | Self::Rgba16 | Self::Gray16 | Self::GrayAlpha16 | Self::Cmyk16
        )
    }

    /// Whether this layout is CMYK (3 colour channels + Black extra
    /// channel). One-shot lossless and lossy are both supported
    /// (lossy passes C/M/Y through XYB via `1-CMY × (1-K)`, K stays
    /// lossless on the Black extra channel); streaming CMYK is not
    /// yet wired.
    pub const fn is_cmyk(self) -> bool {
        matches!(self, Self::Cmyk8 | Self::Cmyk16)
    }

    /// Whether this layout uses f32 samples.
    pub const fn is_f32(self) -> bool {
        matches!(
            self,
            Self::RgbLinearF32
                | Self::RgbaLinearF32
                | Self::GrayLinearF32
                | Self::GrayAlphaLinearF32
                | Self::RgbPqF32
                | Self::RgbaPqF32
                | Self::RgbHlgF32
                | Self::RgbaHlgF32
                | Self::RgbBt709F32
                | Self::RgbaBt709F32
        )
    }

    /// Whether this layout uses IEEE 754 binary16 (f16, half-float)
    /// samples in native-endian u16 storage.
    pub const fn is_f16(self) -> bool {
        matches!(
            self,
            Self::RgbLinearF16
                | Self::RgbaLinearF16
                | Self::GrayLinearF16
                | Self::GrayAlphaLinearF16
        )
    }

    /// Whether this layout includes an alpha channel.
    pub const fn has_alpha(self) -> bool {
        matches!(
            self,
            Self::Rgba8
                | Self::Bgra8
                | Self::GrayAlpha8
                | Self::Rgba16
                | Self::GrayAlpha16
                | Self::RgbaLinearF32
                | Self::GrayAlphaLinearF32
                | Self::RgbaLinearF16
                | Self::GrayAlphaLinearF16
                | Self::RgbaPqF32
                | Self::RgbaHlgF32
                | Self::RgbaBt709F32
        )
    }

    /// Whether this layout is grayscale.
    pub const fn is_grayscale(self) -> bool {
        matches!(
            self,
            Self::Gray8
                | Self::GrayAlpha8
                | Self::Gray16
                | Self::GrayAlpha16
                | Self::GrayLinearF32
                | Self::GrayAlphaLinearF32
                | Self::GrayLinearF16
                | Self::GrayAlphaLinearF16
        )
    }

    /// The transfer function implied by this layout, if any.
    ///
    /// Layouts whose names carry an explicit transfer function (PQ /
    /// HLG / BT.709 f32 input — A3 chunk 1b, issue #46) return
    /// `Some(...)`; sRGB-default and linear layouts return `None`
    /// (caller may still override with [`crate::EncodeRequest::with_color_encoding`] /
    /// [`crate::EncodeRequest::with_color_encoding`]).
    ///
    /// The encoder consults this when the caller did NOT set an
    /// explicit `with_color_encoding(...)` so that PQ / HLG / BT.709
    /// f32 input is signaled correctly in the codestream without
    /// requiring callers to wire a `ColorEncoding` separately.
    pub(crate) fn implied_transfer_function(
        self,
    ) -> Option<crate::headers::color_encoding::TransferFunction> {
        use crate::headers::color_encoding::TransferFunction;
        match self {
            Self::RgbPqF32 | Self::RgbaPqF32 => Some(TransferFunction::Pq),
            Self::RgbHlgF32 | Self::RgbaHlgF32 => Some(TransferFunction::Hlg),
            Self::RgbBt709F32 | Self::RgbaBt709F32 => Some(TransferFunction::Bt709),
            // Linear-named float layouts imply `Linear`. The existing
            // u8 / u16 / sRGB-named layouts do NOT carry an implied TF
            // (caller-controlled via `with_color_encoding`).
            Self::RgbLinearF32
            | Self::RgbaLinearF32
            | Self::GrayLinearF32
            | Self::GrayAlphaLinearF32
            | Self::RgbLinearF16
            | Self::RgbaLinearF16
            | Self::GrayLinearF16
            | Self::GrayAlphaLinearF16 => Some(TransferFunction::Linear),
            _ => None,
        }
    }
}

// ── Quality ─────────────────────────────────────────────────────────────────
