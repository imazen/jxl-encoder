// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! End-to-end "give me HDR + SDR pixels, get one JXL container" entry point.

use alloc::vec::Vec;

use ultrahdr_core::{
    ColorPrimaries as UhdrPrimaries, PixelFormat, TransferFunction as UhdrTransfer, Unstoppable,
    gainmap::{GainMapConfig, compute_gainmap_slice},
    pixel_buffer_from_vec,
};

use crate::api::{EncodeError, LosslessConfig, LossyConfig, PixelLayout};
use crate::headers::color_encoding::ColorEncoding;

use super::bundle::{GainMapBundle, append_gain_map_bundle};

#[path = "from_sdr_color.rs"]
mod color;

pub use ultrahdr_core::GainMapEncodingFormat as GainMapEncoding;

/// Pixel layouts accepted by [`HdrFromSdrRequest`].
///
/// Constrained to the three shapes the gain-map compute kernels support
/// (`Rgba8`, `Rgb8`, `RgbaF32`) on both the SDR and HDR sides. Other
/// shapes (alpha-less HDR f32, BGRA, grayscale, etc.) are not yet wired —
/// upcast on the caller side, or open a feature request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HdrPixelLayout {
    /// 8-bit interleaved RGBA, 4 bytes per pixel.
    Rgba8,
    /// 8-bit interleaved RGB, 3 bytes per pixel.
    Rgb8,
    /// Native-endian IEEE 754 RGBA, 16 bytes per pixel.
    /// Channel order R, G, B, A; the transfer is declared by [`HdrImage::color`].
    RgbaF32,
}

impl HdrPixelLayout {
    /// Bytes per pixel.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 => 3,
            Self::Rgba8 => 4,
            Self::RgbaF32 => 16,
        }
    }

    fn to_jxl_base_layout(self, color: HdrColorEncoding) -> PixelLayout {
        match self {
            Self::Rgba8 => PixelLayout::Rgba8,
            Self::Rgb8 => PixelLayout::Rgb8,
            Self::RgbaF32 => match color.transfer {
                UhdrTransfer::Pq => PixelLayout::RgbaPqF32,
                UhdrTransfer::Hlg(_) => PixelLayout::RgbaHlgF32,
                // sRGB float is linearized privately before this layout is used.
                _ => PixelLayout::RgbaLinearF32,
            },
        }
    }
}

/// Color description for either the SDR or HDR side of an Ultra HDR pair.
///
/// Mirrors the (`primaries`, `transfer`) tuple that
/// `ultrahdr_core::descriptor_for` consumes, with sensible HDR / SDR
/// presets so callers don't have to wrestle with enum aliases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HdrColorEncoding {
    /// Color primaries — `Bt709` for sRGB content, `Bt2020` for HDR.
    pub primaries: UhdrPrimaries,
    /// Transfer function — `Srgb` for SDR base, `Pq` / `Hlg` / `Linear`
    /// for HDR alternates.
    pub transfer: UhdrTransfer,
}

impl HdrColorEncoding {
    /// sRGB (BT.709 primaries + sRGB curve) — the canonical SDR base.
    #[must_use]
    pub const fn srgb() -> Self {
        Self {
            primaries: UhdrPrimaries::Bt709,
            transfer: UhdrTransfer::Srgb,
        }
    }

    /// BT.2100 PQ (BT.2020 primaries + PQ curve) — HDR10 / Ultra HDR
    /// PQ-tagged alternate.
    #[must_use]
    pub const fn bt2100_pq() -> Self {
        Self {
            primaries: UhdrPrimaries::Bt2020,
            transfer: UhdrTransfer::Pq,
        }
    }

    /// BT.2100 HLG (BT.2020 primaries + HLG curve) — broadcast HDR
    /// alternate.
    #[must_use]
    pub const fn bt2100_hlg() -> Self {
        Self {
            primaries: UhdrPrimaries::Bt2020,
            transfer: UhdrTransfer::Hlg(zenpixels_convert::HlgMode::Scene),
        }
    }

    /// Linear RGB tagged with the given primaries — typical for
    /// floating-point HDR sources.
    #[must_use]
    pub const fn linear(primaries: UhdrPrimaries) -> Self {
        Self {
            primaries,
            transfer: UhdrTransfer::Linear,
        }
    }
}

/// One side (SDR or HDR) of an [`HdrFromSdrRequest`] input pair.
///
/// Bundles the byte slice + layout + color encoding so the request
/// constructor takes two short arguments instead of six positional
/// ones (and avoids a clippy `too_many_arguments` lint).
#[derive(Clone, Copy, Debug)]
pub struct HdrImage<'a> {
    /// Raw pixel bytes — tightly packed, length must be
    /// `width * height * layout.bytes_per_pixel()`.
    pub pixels: &'a [u8],
    /// Pixel layout describing channel order and sample type.
    pub layout: HdrPixelLayout,
    /// Color primaries + transfer function the bytes are encoded in.
    pub color: HdrColorEncoding,
}

impl<'a> HdrImage<'a> {
    /// Bundle pixels + layout + color encoding into one side of a
    /// request input pair.
    #[must_use]
    pub const fn new(pixels: &'a [u8], layout: HdrPixelLayout, color: HdrColorEncoding) -> Self {
        Self {
            pixels,
            layout,
            color,
        }
    }
}

/// End-to-end Ultra HDR encode request.
///
/// Given a pair of SDR + HDR pixel buffers (same dimensions, possibly
/// different color encodings), build a single JXL container that:
///
/// 1. Renders the SDR image on standard decoders that ignore the
///    `jhgm` box.
/// 2. Reconstructs the HDR image on capable decoders via
///    `sdr + gainmap → hdr`.
///
/// Gain computation uses the HDR alternate gamut after linearizing both
/// sources. The bundle explicitly declares that alternate color encoding.
/// Both images must have identical alpha; gain maps do not encode alpha changes.
/// Linear and sRGB input light is relative to the base white. PQ and HLG
/// are converted from absolute 10000-nit and 1000-nit display units into that
/// same base white before gain computation. The base header is explicitly bound
/// to its existing 255/1000/10000-nit SDR/HLG/PQ default. A positive alternate
/// intensity target can increase automatic headroom; configured bounds remain.
/// Source conversion must remain finite and nonnegative in the alternate gamut.
/// The supported input descriptions use BT.709, Display P3 or BT.2020 primaries
/// with Linear, sRGB, PQ or HLG transfer; other descriptions return an error.
///
/// The SDR base is encoded via [`LossyConfig`] at the caller's chosen
/// distance. The gain map is encoded via [`LosslessConfig`] (preserves
/// the exact gain values the [`GainMapConfig`] kernel produced).
/// Metadata is serialized via `ultrahdr_core::serialize_iso21496_fmt`
/// in [`Iso21496Format::JxlJhgm`] flavor (no AVIF version byte, no
/// JPEG APP2 URN — that's the wire format `jhgm` boxes expect).
///
/// [`Iso21496Format::JxlJhgm`]: ultrahdr_core::Iso21496Format::JxlJhgm
pub struct HdrFromSdrRequest<'a> {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) sdr_pixels: &'a [u8],
    pub(crate) sdr_layout: HdrPixelLayout,
    pub(crate) sdr_color: HdrColorEncoding,
    pub(crate) hdr_pixels: &'a [u8],
    pub(crate) hdr_layout: HdrPixelLayout,
    pub(crate) hdr_color: HdrColorEncoding,
    pub(crate) hdr_intensity_target: f32,
    pub(crate) lossy_config: LossyConfig,
    pub(crate) gainmap_lossless_config: LosslessConfig,
    pub(crate) gainmap_config: GainMapConfig,
    pub(crate) sdr_color_encoding: Option<ColorEncoding>,
}

impl<'a> HdrFromSdrRequest<'a> {
    /// Construct a new request.
    ///
    /// `sdr_pixels` and `hdr_pixels` must contain exactly
    /// `width * height * <bytes_per_pixel>` bytes each (no stride
    /// field).
    ///
    /// `hdr_intensity_target` is the alternate display peak for automatic
    /// headroom in nits. It must be finite and nonnegative; zero retains the
    /// configured/observed headroom. It does not rescale relative linear input
    /// or change the source HLG interpretation from its 1000-nit display basis.
    ///
    /// Defaults applied:
    /// - SDR JXL distance: `1.0` (use [`Self::with_lossy_config`] to override).
    /// - Gain-map encoding: lossless modular at effort 7.
    /// - Gain-map config: [`GainMapConfig::default()`] (luma gain map,
    ///   1/4 scale, ~2.5 stops boost, `gamma=1.0`).
    /// - SDR base color encoding signaled in JXL: [`ColorEncoding::srgb`].
    ///   Override with [`Self::with_sdr_color_encoding`] if your SDR
    ///   isn't sRGB-tagged.
    pub fn new(
        width: u32,
        height: u32,
        sdr: HdrImage<'a>,
        hdr: HdrImage<'a>,
        hdr_intensity_target: f32,
    ) -> Self {
        Self {
            width,
            height,
            sdr_pixels: sdr.pixels,
            sdr_layout: sdr.layout,
            sdr_color: sdr.color,
            hdr_pixels: hdr.pixels,
            hdr_layout: hdr.layout,
            hdr_color: hdr.color,
            hdr_intensity_target,
            lossy_config: LossyConfig::new(1.0),
            gainmap_lossless_config: LosslessConfig::new(),
            gainmap_config: GainMapConfig::default(),
            sdr_color_encoding: None,
        }
    }

    /// Override the SDR base encoder config (distance, effort, etc.).
    #[must_use]
    pub fn with_lossy_config(mut self, config: LossyConfig) -> Self {
        self.lossy_config = config;
        self
    }

    /// Override the lossless config used to encode the gain-map plane.
    ///
    /// Lossless is the default because gain values are 8-bit ratios —
    /// lossy reconstruction would mangle them. Callers who want a
    /// smaller file can lower the gain-map effort here.
    #[must_use]
    pub fn with_gainmap_lossless_config(mut self, config: LosslessConfig) -> Self {
        self.gainmap_lossless_config = config;
        self
    }

    /// Override the gain-map computation config (scale, gamma, headroom).
    #[must_use]
    pub fn with_gainmap_config(mut self, config: GainMapConfig) -> Self {
        self.gainmap_config = config;
        self
    }

    /// Explicitly signal the SDR base color encoding in the JXL header.
    ///
    /// Defaults to [`ColorEncoding::srgb`] when not set. Pass
    /// [`ColorEncoding::display_p3`] etc. if your SDR base is in a
    /// wide-gamut sRGB-curve space. This must match [`HdrImage::color`];
    /// incompatible overrides return an error rather than reinterpreting samples.
    #[must_use]
    pub fn with_sdr_color_encoding(mut self, ce: ColorEncoding) -> Self {
        self.sdr_color_encoding = Some(ce);
        self
    }

    /// Run the full pipeline and return one JXL container with the
    /// `jhgm` gain-map box appended.
    ///
    /// # Errors
    ///
    /// - [`EncodeError::InvalidInput`] for dimension mismatch, undersized
    ///   pixel buffers, or zero dimensions.
    /// - [`EncodeError::InvalidInput`] (wrapping the inner ultrahdr-core
    ///   error message) if `compute_gainmap` rejects the inputs.
    /// - Whatever error the inner JXL encode raises (propagated as-is).
    pub fn encode(self) -> Result<Vec<u8>, EncodeError> {
        // 1. Validate input buffers.
        validate_buffer(
            self.sdr_pixels,
            self.width,
            self.height,
            self.sdr_layout,
            "sdr",
        )?;
        validate_buffer(
            self.hdr_pixels,
            self.width,
            self.height,
            self.hdr_layout,
            "hdr",
        )?;

        // Compute in the declared alternate gamut after each source EOTF.
        // Normalizing every layout also avoids the producer's 8-bit fallback
        // to sRGB for explicitly declared PQ/HLG sources.
        let sdr_ce = color::source_override(
            self.sdr_color_encoding.clone(),
            color::encoding(self.sdr_color)?,
        )?;
        let hdr_ce = color::encoding(self.hdr_color)?;
        if !self.hdr_intensity_target.is_finite() || self.hdr_intensity_target < 0.0 {
            return Err(invalid(
                "hdr-gainmap: alternate intensity target must be finite and nonnegative".into(),
            ));
        }
        let base_white = color::base_white_nits(self.sdr_color);
        let gain_space = color::GainSpace {
            primaries: self.hdr_color.primaries,
            white_nits: base_white,
        };
        let sdr_linear = color::linear_rgba(
            self.sdr_pixels,
            self.width,
            self.height,
            self.width as usize * self.sdr_layout.bytes_per_pixel(),
            self.sdr_layout,
            self.sdr_color,
            gain_space,
        )?;
        let hdr_linear = color::linear_rgba(
            self.hdr_pixels,
            self.width,
            self.height,
            self.width as usize * self.hdr_layout.bytes_per_pixel(),
            self.hdr_layout,
            self.hdr_color,
            gain_space,
        )?;
        let normalized_stride = self.width as usize * 16;
        for (sdr, hdr) in sdr_linear
            .chunks_exact(normalized_stride)
            .zip(hdr_linear.chunks_exact(normalized_stride))
        {
            color::matching_alpha_row(sdr, hdr)?;
        }
        let sdr_buf = pixel_buffer_from_vec(
            sdr_linear,
            self.width,
            self.height,
            PixelFormat::RGBAF32,
            self.hdr_color.primaries,
            UhdrTransfer::Linear,
        )
        .map_err(uhdr_error)?;
        let hdr_buf = pixel_buffer_from_vec(
            hdr_linear,
            self.width,
            self.height,
            PixelFormat::RGBAF32,
            self.hdr_color.primaries,
            UhdrTransfer::Linear,
        )
        .map_err(uhdr_error)?;

        // 3. Compute the gain map.
        let (gainmap, mut metadata) = compute_gainmap_slice(
            hdr_buf.as_view(),
            sdr_buf.as_view(),
            &self.gainmap_config,
            Unstoppable,
        )
        .map_err(uhdr_error)?;
        // Gain computation no longer borrows its normalized input planes.
        drop(sdr_buf);
        drop(hdr_buf);
        metadata.use_base_color_space = false;

        // 4. Headroom uses the same base-white units as the gain ratios.
        // Configured quantization bounds and stricter headroom remain intact.
        // Zero retains the compute kernel's configured/observed headroom.
        if self.hdr_intensity_target > 0.0 {
            let target_log2 = (f64::from(self.hdr_intensity_target) / f64::from(base_white))
                .log2()
                .max(0.0);
            metadata.alternate_hdr_headroom = metadata.alternate_hdr_headroom.max(target_log2);
        }

        // 5. Encode the SDR base as a JXL codestream. We thread the
        //    caller-chosen color encoding via the public request layer
        //    so the bytes the SDR decoder gets are correctly tagged.
        // The public float layout declares its input transfer. PQ/HLG have
        // typed base layouts; nonlinear sRGB float needs linear preparation.
        let mut prepared_sdr = if self.sdr_layout == HdrPixelLayout::RgbaF32
            && self.sdr_color.transfer == UhdrTransfer::Srgb
        {
            Some(color::linear_rgba(
                self.sdr_pixels,
                self.width,
                self.height,
                self.width as usize * 16,
                self.sdr_layout,
                self.sdr_color,
                color::GainSpace {
                    primaries: self.sdr_color.primaries,
                    white_nits: base_white,
                },
            )?)
        } else {
            None
        };
        if let Some(pixels) = &mut prepared_sdr {
            // The gain producer reads little-endian floats; the pixel API
            // accepts native-endian floats. This conversion is a no-op on LE.
            for sample in pixels.as_chunks_mut::<4>().0 {
                *sample = f32::from_le_bytes(*sample).to_ne_bytes();
            }
        }
        let sdr_request = self
            .lossy_config
            .encode_request(
                self.width,
                self.height,
                self.sdr_layout.to_jxl_base_layout(self.sdr_color),
            )
            .with_color_encoding(sdr_ce)
            .with_intensity_target(base_white);
        let sdr_jxl = sdr_request
            .encode(prepared_sdr.as_deref().unwrap_or(self.sdr_pixels))
            .map_err(at_to_inner)?;

        // 6. Encode the gain-map plane as a lossless JXL codestream.
        // Single-channel grayscale for luminance gain maps; 3-channel
        // RGB for per-channel gain maps. ultrahdr-core fills `data`
        // tightly-packed in either case.
        let (gm_layout, _gm_bpp) = if gainmap.channels == 1 {
            (PixelLayout::Gray8, 1usize)
        } else {
            (PixelLayout::Rgb8, 3usize)
        };
        let gm_jxl_request =
            self.gainmap_lossless_config
                .encode_request(gainmap.width, gainmap.height, gm_layout);
        let gm_jxl = gm_jxl_request.encode(&gainmap.data).map_err(at_to_inner)?;

        // 7. Serialize the ISO 21496-1 metadata for the `jhgm` payload.
        let iso21496 = ultrahdr_core::serialize_iso21496_fmt(
            &metadata,
            ultrahdr_core::Iso21496Format::JxlJhgm,
        )
        .map_err(|err| invalid(format!("hdr-gainmap: {err}")))?;

        // Part 2 9.11: alternate color describes reconstructed HDR samples;
        // gain values themselves retain their separate lossless coding.
        let bundle = GainMapBundle::new(iso21496, gm_jxl).with_color_encoding(hdr_ce);
        let container = append_gain_map_bundle(&sdr_jxl, &bundle)?;
        Ok(container)
    }
}

/// Adapt the traced Ultra HDR error into our `EncodeError`. The variants
/// don't line up 1:1, so we route everything through `InvalidInput`
/// with the original error text — the alternative would be to extend
/// `EncodeError` with HDR-specific variants which would leak even when
/// the `hdr-gainmap` feature is off.
fn uhdr_error(e: crate::api::At<ultrahdr_core::Error>) -> EncodeError {
    // ultrahdr-core 0.6 returns whereat-wrapped errors; keep the message text.
    invalid(format!("hdr-gainmap: {}", e.error()))
}

fn at_to_inner(at: crate::api::At<EncodeError>) -> EncodeError {
    // Drop the source-location wrapper; callers of HdrFromSdrRequest
    // don't need it (the `?` site here is the only relevant one).
    // `decompose()` is the non-deprecated way to recover the inner
    // error without losing any state we care about.
    let (err, _trace) = at.decompose();
    err
}

/// Helper: build an `EncodeError::InvalidInput` from a `String`.
#[inline]
fn invalid(message: alloc::string::String) -> EncodeError {
    EncodeError::InvalidInput { message }
}

fn validate_buffer(
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: HdrPixelLayout,
    side: &str,
) -> Result<(), EncodeError> {
    if width == 0 || height == 0 {
        return Err(invalid(format!(
            "hdr-gainmap: zero dimensions ({width}x{height})"
        )));
    }
    let bpp = layout.bytes_per_pixel();
    let need = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(bpp))
        .ok_or_else(|| {
            invalid(format!(
                "hdr-gainmap: dimension overflow {width}x{height}x{bpp}"
            ))
        })?;
    if pixels.len() < need {
        return Err(invalid(format!(
            "hdr-gainmap: {side} buffer too small: {} bytes, need {} (= {}x{}x{})",
            pixels.len(),
            need,
            width,
            height,
            bpp
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: 8×8 SDR + HDR Rgba8 pair → JXL container with `jhgm`.
    #[test]
    fn small_pair_produces_container_with_jhgm() {
        let w = 8u32;
        let h = 8u32;
        // SDR: mid-gray everywhere (128).
        let mut sdr = vec![128u8; (w * h * 4) as usize];
        for pixel in sdr.as_chunks_mut::<4>().0 {
            pixel[3] = 255;
        }
        // HDR: same RGB but with each pixel slightly above SDR so the
        // gain map has something to encode. Without any signal the
        // gainmap kernel still produces a valid bundle, but this is
        // closer to a real Ultra HDR pair.
        let mut hdr = vec![0u8; (w * h * 4) as usize];
        for (i, p) in hdr.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let bright = 128u8.saturating_add((i % 64) as u8);
            p[0] = bright;
            p[1] = bright;
            p[2] = bright;
            p[3] = 255;
        }

        let req = HdrFromSdrRequest::new(
            w,
            h,
            HdrImage::new(&sdr, HdrPixelLayout::Rgba8, HdrColorEncoding::srgb()),
            HdrImage::new(&hdr, HdrPixelLayout::Rgba8, HdrColorEncoding::srgb()),
            1000.0,
        );
        let bytes = req.encode().expect("encode HDR-from-SDR");

        // Must be a JXL container.
        assert_eq!(
            &bytes[..12],
            &[
                0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A
            ],
            "expected JXL container signature"
        );
        // libjxl's layout: the SDR base codestream split after its header,
        // with the `jhgm` box between the header and the frames.
        assert_eq!(
            crate::container::box_types(&bytes),
            [*b"JXL ", *b"ftyp", *b"jxlp", *b"jhgm", *b"jxlp"]
        );
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let sdr = vec![0u8; 8 * 8 * 4];
        let hdr = vec![0u8; 4 * 4 * 4]; // too small
        let req = HdrFromSdrRequest::new(
            8,
            8,
            HdrImage::new(&sdr, HdrPixelLayout::Rgba8, HdrColorEncoding::srgb()),
            HdrImage::new(&hdr, HdrPixelLayout::Rgba8, HdrColorEncoding::bt2100_pq()),
            1000.0,
        );
        let err = req.encode().expect_err("undersized HDR buffer should fail");
        assert!(matches!(err, EncodeError::InvalidInput { .. }));
    }

    #[test]
    fn zero_dimensions_rejected() {
        let req = HdrFromSdrRequest::new(
            0,
            0,
            HdrImage::new(&[], HdrPixelLayout::Rgba8, HdrColorEncoding::srgb()),
            HdrImage::new(&[], HdrPixelLayout::Rgba8, HdrColorEncoding::bt2100_pq()),
            1000.0,
        );
        assert!(req.encode().is_err());
    }
}
