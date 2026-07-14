// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Embeddable image metadata (\`ImageMetadata\`): ICC, EXIF, XMP, JUMBF, tone mapping.

/// Image metadata (ICC, EXIF, XMP, JUMBF, tone mapping) to embed in the JXL file.
#[derive(Clone, Debug, Default)]
pub struct ImageMetadata<'a> {
    pub(crate) icc_profile: Option<&'a [u8]>,
    pub(crate) exif: Option<&'a [u8]>,
    pub(crate) xmp: Option<&'a [u8]>,
    /// JUMBF (JPEG Universal Metadata Box Format, ISO 19566-5) payload,
    /// emitted as a `jumb` ISOBMFF box appended after `Exif`/`xml `.
    /// Used by C2PA / Content Authenticity Initiative tooling. The
    /// encoder passes the bytes through verbatim — no validation.
    pub(crate) jumbf: Option<&'a [u8]>,
    /// Alternative colour-descriptor box payload (ISOBMFF `colr`, ISO/IEC
    /// 14496-12), appended after all other metadata boxes. Pass-through
    /// only — the encoder does not interpret. Use
    /// [`crate::container::colr_nclx_payload`] to build a conformant nclx
    /// payload from CICP enum values.
    pub(crate) colr_payload: Option<&'a [u8]>,
    /// HDR content-description box payload (`hCdR`), appended after all
    /// other metadata boxes. Pass-through only — caller assembles the
    /// schema-specific bytes (e.g. SMPTE ST 2086 + CTA-861.3
    /// MaxCLL/MaxFALL). The encoder does not validate.
    pub(crate) hcdr_payload: Option<&'a [u8]>,
    /// Peak display luminance in nits (cd/m²). `None` uses the JXL default (255.0 = SDR).
    pub(crate) intensity_target: Option<f32>,
    /// Minimum display luminance in nits. `None` uses the JXL default (0.0).
    pub(crate) min_nits: Option<f32>,
    /// `ToneMapping.relative_to_max_display` (default `false`). `None`
    /// uses the JXL default. Issue #46 chunk 1a.
    pub(crate) relative_to_max_display: Option<bool>,
    /// `ToneMapping.linear_below` (default `0.0`). `None` uses the JXL
    /// default. Interpretation depends on
    /// [`Self::relative_to_max_display`] (ratio when `true`, absolute
    /// nits when `false`). Issue #46 chunk 1a.
    pub(crate) linear_below: Option<f32>,
    /// Intrinsic display size `(width, height)`, if different from coded dimensions.
    pub(crate) intrinsic_size: Option<(u32, u32)>,
}

impl<'a> ImageMetadata<'a> {
    /// Create empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach an ICC color profile.
    pub fn with_icc_profile(mut self, data: &'a [u8]) -> Self {
        self.icc_profile = Some(data);
        self
    }

    /// Attach EXIF data.
    pub fn with_exif(mut self, data: &'a [u8]) -> Self {
        self.exif = Some(data);
        self
    }

    /// Attach XMP data.
    pub fn with_xmp(mut self, data: &'a [u8]) -> Self {
        self.xmp = Some(data);
        self
    }

    /// Attach a JUMBF (JPEG Universal Metadata Box Format) payload.
    ///
    /// The bytes are written verbatim into a `jumb` ISOBMFF box appended
    /// after the standard `Exif`/`xml ` boxes. Used by C2PA / Content
    /// Authenticity Initiative tooling for provenance metadata; the
    /// caller produces the JUMBF superbox (typically via the `c2pa`
    /// crate) and we pass it through without inspection. Mirrors
    /// libjxl's `JxlEncoderAddBox(enc, "jumb", ...)` API.
    pub fn with_jumbf(mut self, data: &'a [u8]) -> Self {
        self.jumbf = Some(data);
        self
    }

    /// Get the ICC color profile, if set.
    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.icc_profile
    }

    /// Get the EXIF data, if set.
    pub fn exif(&self) -> Option<&[u8]> {
        self.exif
    }

    /// Get the XMP data, if set.
    pub fn xmp(&self) -> Option<&[u8]> {
        self.xmp
    }

    /// Get the JUMBF payload, if set.
    pub fn jumbf(&self) -> Option<&[u8]> {
        self.jumbf
    }

    /// Attach an alternative colour-descriptor box (`colr`,
    /// ISO/IEC 14496-12 ColourInformationBox).
    ///
    /// `data` is the raw box content — the first 4 bytes are the
    /// `colour_type` FourCC (`nclx`, `rICC`, `prof`, …) and the rest is
    /// the subtype-specific payload. Use
    /// [`crate::container::colr_nclx_payload`] to construct an nclx
    /// payload from CICP enum values.
    ///
    /// JPEG XL signals its primary colour information in-codestream via
    /// [`crate::headers::color_encoding::ColourEncoding`]; this box is
    /// an **alternative descriptor** for ISOBMFF-aware tooling
    /// (HEIF/AVIF metadata inspectors). Per JPEG XL spec clause 5,
    /// decoders MUST ignore boxes with unrecognised types — so emitting
    /// this box never alters decoded pixels.
    ///
    /// Only honoured on the one-shot [`EncodeRequest`] path. Streaming
    /// encoders (`LossyEncoder` / `LosslessEncoder`) do not surface
    /// `ImageMetadata` and silently drop this field.
    pub fn with_colr_payload(mut self, data: &'a [u8]) -> Self {
        self.colr_payload = Some(data);
        self
    }

    /// Get the `colr` payload, if set.
    pub fn colr_payload(&self) -> Option<&[u8]> {
        self.colr_payload
    }

    /// Attach an HDR content-description box (`hCdR`) payload.
    ///
    /// `data` is the raw box content. The encoder does not validate or
    /// interpret it — callers assemble the schema-specific bytes for
    /// their downstream tooling (e.g. SMPTE ST 2086 mastering display
    /// volume + CTA-861.3 MaxCLL/MaxFALL).
    ///
    /// JPEG XL signals peak/min display luminance in-codestream via
    /// [`crate::headers::color_encoding::ToneMapping`]
    /// (`intensity_target`, `min_nits`). This box is an **alternative
    /// descriptor** for ISOBMFF-aware HDR tooling. Per JPEG XL spec
    /// clause 5, decoders MUST ignore boxes with unrecognised types —
    /// so emitting this box never alters decoded pixels.
    ///
    /// Only honoured on the one-shot [`EncodeRequest`] path. Streaming
    /// encoders (`LossyEncoder` / `LosslessEncoder`) do not surface
    /// `ImageMetadata` and silently drop this field.
    pub fn with_hcdr_payload(mut self, data: &'a [u8]) -> Self {
        self.hcdr_payload = Some(data);
        self
    }

    /// Get the `hCdR` payload, if set.
    pub fn hcdr_payload(&self) -> Option<&[u8]> {
        self.hcdr_payload
    }

    /// Set the peak display luminance in nits (cd/m²) for HDR content.
    ///
    /// Written to the JXL codestream `ToneMapping.intensity_target` field.
    /// Default is 255.0 (SDR). Set to e.g. 4000.0 or 10000.0 for HDR.
    pub fn with_intensity_target(mut self, nits: f32) -> Self {
        self.intensity_target = Some(nits);
        self
    }

    /// Set the minimum display luminance in nits.
    ///
    /// Written to the JXL codestream `ToneMapping.min_nits` field.
    /// Default is 0.0.
    pub fn with_min_nits(mut self, nits: f32) -> Self {
        self.min_nits = Some(nits);
        self
    }

    /// Set `ToneMapping.relative_to_max_display`.
    ///
    /// When `true`, [`Self::with_linear_below`] is interpreted as a
    /// ratio in `[0, 1]` of the maximum display brightness. When
    /// `false` (the default), it is an absolute nit value. Mirrors
    /// libjxl `JxlBasicInfo::relative_to_max_display`. Closes issue
    /// #46 chunk 1a.
    pub fn with_relative_to_max_display(mut self, relative: bool) -> Self {
        self.relative_to_max_display = Some(relative);
        self
    }

    /// Set `ToneMapping.linear_below`.
    ///
    /// Tone-mapping leaves pixels strictly below this value unchanged
    /// (linear). Default is `0.0` (always tone-map). Interpretation
    /// depends on [`Self::with_relative_to_max_display`] — when
    /// `true`, this is a ratio in `[0, 1]`; otherwise an absolute nit
    /// value. Mirrors libjxl `JxlBasicInfo::linear_below`. Closes
    /// issue #46 chunk 1a.
    pub fn with_linear_below(mut self, value: f32) -> Self {
        self.linear_below = Some(value);
        self
    }

    /// Get the intensity target, if set.
    pub fn intensity_target(&self) -> Option<f32> {
        self.intensity_target
    }

    /// Get the min nits, if set.
    pub fn min_nits(&self) -> Option<f32> {
        self.min_nits
    }

    /// Get the `relative_to_max_display` flag, if set.
    pub fn relative_to_max_display(&self) -> Option<bool> {
        self.relative_to_max_display
    }

    /// Get the `linear_below` value, if set.
    pub fn linear_below(&self) -> Option<f32> {
        self.linear_below
    }

    /// Set the intrinsic display size.
    ///
    /// When set, the image should be rendered at this `(width, height)` rather
    /// than the coded dimensions. Written to the JXL codestream `intrinsic_size` field.
    pub fn with_intrinsic_size(mut self, width: u32, height: u32) -> Self {
        self.intrinsic_size = Some((width, height));
        self
    }

    /// Get the intrinsic size, if set.
    pub fn intrinsic_size(&self) -> Option<(u32, u32)> {
        self.intrinsic_size
    }
}
