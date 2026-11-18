// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//! Color preparation shared by all packed HDR pair input layouts.
use super::{HdrColorEncoding, HdrPixelLayout, invalid};
use crate::api::EncodeError;
use crate::headers::color_encoding::ColorEncoding;
use alloc::vec::Vec;
use ultrahdr_core::{ColorPrimaries, TransferFunction};

pub(super) fn encoding(color: HdrColorEncoding) -> Result<ColorEncoding, EncodeError> {
    let cp = match color.primaries {
        ColorPrimaries::Bt709 => 1,
        ColorPrimaries::Bt2020 => 9,
        ColorPrimaries::DisplayP3 => 12,
        _ => return Err(invalid("hdr-gainmap: unsupported source primaries".into())),
    };
    let tc = match color.transfer {
        TransferFunction::Linear => 8,
        TransferFunction::Srgb => 13,
        TransferFunction::Pq => 16,
        TransferFunction::Hlg => 18,
        _ => {
            return Err(invalid(
                "hdr-gainmap: unsupported source transfer function".into(),
            ));
        }
    };
    ColorEncoding::from_cicp(cp, tc, 0, true).map_err(|message| invalid(message.into()))
}

pub(super) fn source_override(
    supplied: Option<ColorEncoding>,
    expected: ColorEncoding,
) -> Result<ColorEncoding, EncodeError> {
    // Preserve the public request's documented sRGB default. A differently
    // described SDR source needs its matching explicit encoding override.
    let supplied = supplied.unwrap_or_else(ColorEncoding::srgb);
    if supplied.color_space != expected.color_space
        || supplied.want_icc
        || supplied.white_point != expected.white_point
        || supplied.primaries != expected.primaries
        || supplied.transfer_function != expected.transfer_function
        || supplied.gamma != expected.gamma
        || supplied.custom_white_point != expected.custom_white_point
        || supplied.custom_primaries != expected.custom_primaries
    {
        return Err(invalid(
            "hdr-gainmap: SDR default or override is unsupported or disagrees with the input color description".into(),
        ));
    }
    Ok(supplied)
}

/// Common gain-space units: linear RGB 1.0 means the base image's white.
#[derive(Clone, Copy)]
pub(super) struct GainSpace {
    pub(super) primaries: ColorPrimaries,
    pub(super) white_nits: f32,
}

/// Bind the base header's peak explicitly instead of assuming a separate
/// diffuse-white reference. These are the existing typed-input defaults.
pub(super) fn base_white_nits(color: HdrColorEncoding) -> f32 {
    match color.transfer {
        TransferFunction::Pq => 10_000.0,
        TransferFunction::Hlg => 1_000.0,
        _ => 255.0,
    }
}

/// Prepare linear RGBA in the selected gain gamut. Row padding is never read.
/// PQ is absolute relative to 10000 nits. HLG retains the producer's 1000-nit
/// display interpretation, including its luminance-based OOTF. Convert both
/// to the base white; plain linear/sRGB samples are already relative to it.
pub(super) fn linear_rgba(
    pixels: &[u8],
    width: u32,
    rows: u32,
    stride: usize,
    layout: HdrPixelLayout,
    color: HdrColorEncoding,
    target: GainSpace,
) -> Result<Vec<u8>, EncodeError> {
    let source_encoding = encoding(color)?;
    if !target.white_nits.is_finite() || target.white_nits <= 0.0 {
        return Err(invalid(
            "hdr-gainmap: base white must be finite and positive".into(),
        ));
    }
    let light_scale = match color.transfer {
        TransferFunction::Pq => 10_000.0 / target.white_nits,
        TransferFunction::Hlg => 1_000.0 / target.white_nits,
        _ => 1.0,
    };
    let hlg = (color.transfer == TransferFunction::Hlg).then(|| {
        (
            crate::api::hlg_ootf_luminances(source_encoding.primaries),
            crate::api::hlg_ootf_gamma(1000.0).expect("1000-nit HLG has non-unity system gamma"),
        )
    });
    let width_usize = width as usize;
    let bpp = layout.bytes_per_pixel();
    let row_bytes = width_usize
        .checked_mul(bpp)
        .ok_or_else(|| invalid("hdr-gainmap: source row size overflows".into()))?;
    let extent = (rows as usize)
        .checked_sub(1)
        .and_then(|r| r.checked_mul(stride))
        .and_then(|offset| offset.checked_add(row_bytes));
    if width == 0 || stride < row_bytes || extent.is_none_or(|extent| pixels.len() < extent) {
        return Err(invalid("hdr-gainmap: incomplete source rows".into()));
    }
    let output_stride = width_usize
        .checked_mul(16)
        .ok_or_else(|| invalid("hdr-gainmap: output row size overflows".into()))?;
    let length = output_stride
        .checked_mul(rows as usize)
        .ok_or_else(|| invalid("hdr-gainmap: output image size overflows".into()))?;
    let matrix = if color.primaries == target.primaries {
        None
    } else {
        Some(
            color
                .primaries
                .gamut_matrix_to(target.primaries)
                .ok_or_else(|| {
                    invalid(
                        "hdr-gainmap: source-to-alternate gamut conversion is unavailable".into(),
                    )
                })?,
        )
    };
    let mut output =
        crate::budget::vec_with_capacity_fallible::<u8>(true, length).map_err(EncodeError::from)?;
    output.resize(length, 0);
    let mut planes =
        crate::budget::vec_f32_zeroed_fallible(true, width_usize * 3).map_err(EncodeError::from)?;
    for y in 0..rows as usize {
        let source = &pixels[y * stride..][..row_bytes];
        let dest = &mut output[y * output_stride..][..output_stride];
        let (red, gb) = planes.split_at_mut(width_usize);
        let (green, blue) = gb.split_at_mut(width_usize);
        for x in 0..width_usize {
            let bytes = &source[x * bpp..][..bpp];
            let mut rgba = match layout {
                HdrPixelLayout::Rgb8 => [
                    f32::from(bytes[0]) / 255.0,
                    f32::from(bytes[1]) / 255.0,
                    f32::from(bytes[2]) / 255.0,
                    1.0,
                ],
                HdrPixelLayout::Rgba8 => [
                    f32::from(bytes[0]) / 255.0,
                    f32::from(bytes[1]) / 255.0,
                    f32::from(bytes[2]) / 255.0,
                    f32::from(bytes[3]) / 255.0,
                ],
                HdrPixelLayout::RgbaF32 => core::array::from_fn(|c| {
                    f32::from_ne_bytes(bytes[c * 4..c * 4 + 4].try_into().unwrap())
                }),
            };
            if rgba.iter().any(|v| !v.is_finite()) {
                return Err(invalid("hdr-gainmap: non-finite source sample".into()));
            }
            for value in &mut rgba[..3] {
                *value = match color.transfer {
                    TransferFunction::Linear => *value,
                    TransferFunction::Srgb => ultrahdr_core::color::transfer::srgb_eotf(*value),
                    TransferFunction::Pq => ultrahdr_core::color::transfer::pq_eotf(*value),
                    TransferFunction::Hlg => crate::api::hlg_to_linear_f(*value),
                    _ => unreachable!("validated transfer function"),
                };
            }
            if let Some((luminances, gamma)) = hlg {
                crate::api::apply_hlg_forward_ootf(&mut rgba[..3], luminances, gamma);
            }
            if light_scale != 1.0 {
                for value in &mut rgba[..3] {
                    *value *= light_scale;
                }
            }
            red[x] = rgba[0];
            green[x] = rgba[1];
            blue[x] = rgba[2];
            dest[x * 16 + 12..x * 16 + 16].copy_from_slice(&rgba[3].to_le_bytes());
        }
        if let Some(matrix) = &matrix {
            crate::vardct::xyb::apply_matrix_3x3(red, green, blue, matrix);
        }
        for x in 0..width_usize {
            for (c, value) in [red[x], green[x], blue[x]].into_iter().enumerate() {
                if !value.is_finite() || value < 0.0 {
                    return Err(invalid("hdr-gainmap: negative or non-finite RGB after alternate gamut conversion is not supported".into()));
                }
                dest[x * 16 + c * 4..x * 16 + c * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(output)
}

/// A gain map changes color, while alternate alpha is inherited from the base.
/// Compare one normalized RGBA row without reading any caller padding.
pub(super) fn matching_alpha_row(sdr: &[u8], hdr: &[u8]) -> Result<(), EncodeError> {
    if sdr.len() != hdr.len() || !sdr.len().is_multiple_of(16) {
        return Err(invalid(
            "hdr-gainmap: incomplete normalized alpha row".into(),
        ));
    }
    for (sdr, hdr) in sdr.as_chunks::<16>().0.iter().zip(hdr.as_chunks::<16>().0) {
        let alpha = f32::from_le_bytes(sdr[12..16].try_into().unwrap());
        if !(0.0..=1.0).contains(&alpha) || sdr[12..16] != hdr[12..16] {
            return Err(invalid(
                "hdr-gainmap: alternate alpha must match the base exactly within [0, 1]".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "from_sdr_color_tests.rs"]
mod tests;
