// Copyright (c) Imazen LLC. Licensed under AGPL-3.0-or-later.
//! Profile-based CMYK interpretation before the shared linear-sRGB pipeline.
use crate::{
    EncodeError,
    budget::{BudgetGuard, MemoryBudget},
};
use alloc::{sync::Arc, vec::Vec};
use moxcms::{ColorProfile, DataColorSpace, Layout, RenderingIntent, TransformOptions};

#[derive(Clone, Copy)]
pub(super) enum Black<'a> {
    U8(&'a [u8], f32),
    U16(&'a [u16], f32),
    Interleaved,
}
pub(super) struct InkRows<'a> {
    pub pixels: &'a [u8],
    pub width: usize,
    pub rows: usize,
    pub stride: usize,
    pub channels: usize,
    pub bytes_per_sample: usize,
    pub maximum: f32,
    pub order: [usize; 3],
    pub black: Black<'a>,
}
pub(super) fn to_linear_srgb(
    input: InkRows<'_>,
    profile: &[u8],
    budget: &Arc<MemoryBudget>,
    fallible: bool,
    stop: Option<&dyn enough::Stop>,
) -> Result<(Vec<f32>, BudgetGuard), EncodeError> {
    let invalid = |message: &str| EncodeError::InvalidInput {
        message: message.into(),
    };
    let pixel_bytes = input
        .channels
        .checked_mul(input.bytes_per_sample)
        .ok_or_else(|| invalid("CMYK pixel size overflows"))?;
    let row_bytes = input
        .width
        .checked_mul(pixel_bytes)
        .ok_or_else(|| invalid("CMYK row size overflows"))?;
    let count = input
        .width
        .checked_mul(input.rows)
        .ok_or_else(|| invalid("CMYK image size overflows"))?;
    let length = count
        .checked_mul(3)
        .ok_or_else(|| invalid("CMYK output size overflows"))?;
    let extent = input
        .rows
        .checked_sub(1)
        .and_then(|n| n.checked_mul(input.stride))
        .and_then(|n| n.checked_add(row_bytes))
        .ok_or_else(|| invalid("CMYK source extent overflows"))?;
    if !matches!(input.bytes_per_sample, 1 | 2)
        || input.channels < 3
        || input.width == 0
        || input.pixels.len() < extent
        || input.stride < row_bytes
        || input.order.iter().any(|&channel| channel >= input.channels)
        || !input.maximum.is_finite()
        || input.maximum <= 0.0
    {
        return Err(invalid("CMYK source rows are invalid"));
    }
    match input.black {
        Black::U8(values, maximum)
            if values.len() != count || !maximum.is_finite() || maximum <= 0.0 =>
        {
            return Err(invalid("CMYK Black samples are invalid"));
        }
        Black::U16(values, maximum)
            if values.len() != count || !maximum.is_finite() || maximum <= 0.0 =>
        {
            return Err(invalid("CMYK Black samples are invalid"));
        }
        Black::Interleaved if input.channels < 4 => {
            return Err(invalid("CMYK needs four interleaved channels"));
        }
        _ => {}
    }
    crate::error::check_stop(stop)?;
    let source = ColorProfile::new_from_slice(profile)
        .map_err(|error| invalid(&format!("CMYK ICC: {error}")))?;
    if source.color_space != DataColorSpace::Cmyk {
        return Err(invalid("CMYK conversion requires a CMYK ICC profile"));
    }
    let mut target = ColorProfile::new_srgb();
    let curve = moxcms::curve_from_gamma(1.0);
    target.red_trc = Some(curve.clone());
    target.green_trc = Some(curve.clone());
    target.blue_trc = Some(curve);
    target.cicp = None;
    // LCMS's relative-colorimetric control uses A2B0 if A2B1 is absent.
    // The pinned moxcms selector requires the available table explicitly.
    let options = TransformOptions {
        rendering_intent: if source.lut_a_to_b_colorimetric.is_some() {
            RenderingIntent::RelativeColorimetric
        } else {
            RenderingIntent::Perceptual
        },
        allow_use_cicp_transfer: false,
        prefer_fixed_point: false,
        interpolation_method: moxcms::InterpolationMethod::Tetrahedral,
        allow_extended_range_rgb_xyz: true,
        ..Default::default()
    };
    let transform = source
        .create_transform_f32(Layout::Rgba, &target, Layout::Rgb, options)
        .map_err(|error| invalid(&format!("CMYK profile conversion: {error}")))?;
    let output_bytes = length
        .checked_mul(4)
        .ok_or_else(|| invalid("CMYK output bytes overflow"))?;
    let guard =
        MemoryBudget::reserve_opt(Some(budget), output_bytes as u64).map_err(EncodeError::from)?;
    let mut output =
        crate::budget::vec_f32_zeroed_fallible(fallible, length).map_err(EncodeError::from)?;
    let scratch_len = input
        .width
        .checked_mul(4)
        .ok_or_else(|| invalid("CMYK scratch size overflows"))?;
    let scratch_bytes = scratch_len
        .checked_mul(4)
        .ok_or_else(|| invalid("CMYK scratch bytes overflow"))?;
    let _scratch_guard =
        MemoryBudget::reserve_opt(Some(budget), scratch_bytes as u64).map_err(EncodeError::from)?;
    let mut scratch =
        crate::budget::vec_f32_zeroed_fallible(fallible, scratch_len).map_err(EncodeError::from)?;
    for row in 0..input.rows {
        crate::error::check_stop(stop)?;
        let bytes = &input.pixels[row * input.stride..][..row_bytes];
        for (x, target) in scratch.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let pixel = &bytes[x * pixel_bytes..][..pixel_bytes];
            let sample = |channel: usize| {
                let i = channel * input.bytes_per_sample;
                if input.bytes_per_sample == 1 {
                    f32::from(pixel[i])
                } else {
                    f32::from(u16::from_ne_bytes([pixel[i], pixel[i + 1]]))
                }
            };
            for (value, &channel) in target[..3].iter_mut().zip(&input.order) {
                *value = 1.0 - sample(channel) / input.maximum;
            }
            let index = row * input.width + x;
            target[3] = 1.0
                - match input.black {
                    Black::U8(values, max) => f32::from(values[index]) / max,
                    Black::U16(values, max) => f32::from(values[index]) / max,
                    Black::Interleaved => sample(3) / input.maximum,
                };
            if target
                .iter()
                .any(|&v| !v.is_finite() || !(0.0..=1.0).contains(&v))
            {
                return Err(invalid("CMYK samples exceed their declared precision"));
            }
        }
        let target = &mut output[row * input.width * 3..][..input.width * 3];
        transform
            .transform(&scratch, target)
            .map_err(|error| invalid(&format!("CMYK row conversion: {error}")))?;
        if target.iter().any(|v| !v.is_finite()) {
            return Err(invalid("CMYK conversion produced non-finite light"));
        }
    }
    Ok((output, guard))
}

impl super::EncodeRequest<'_> {
    pub(super) fn prepare_cmyk_pixels(
        &self,
        pixels: &[u8],
        width: usize,
        rows: usize,
        budget: &Arc<MemoryBudget>,
    ) -> Result<super::LossyPixels, EncodeError> {
        use super::{ExtraChannelBuf, PixelLayout};
        let invalid = |message: &str| EncodeError::InvalidInput {
            message: message.into(),
        };
        if !matches!(
            self.layout,
            PixelLayout::Rgb8
                | PixelLayout::Rgba8
                | PixelLayout::Bgr8
                | PixelLayout::Bgra8
                | PixelLayout::Rgb16
                | PixelLayout::Rgba16
                | PixelLayout::Cmyk8
                | PixelLayout::Cmyk16
        ) {
            return Err(invalid(
                "CMYK profile conversion needs integer RGB(A), BGR(A) or CMYK samples",
            ));
        }
        if width == 0 || rows == 0 {
            return Err(invalid("CMYK source dimensions must be nonzero"));
        }
        if self
            .bits_per_sample
            .is_some_and(|bits| !(1..=16).contains(&bits))
        {
            return Err(invalid("CMYK integer precision must be 1..=16"));
        }
        let count = width
            .checked_mul(rows)
            .ok_or_else(|| invalid("CMYK source size overflows"))?;
        let expected = count
            .checked_mul(self.layout.bytes_per_pixel())
            .ok_or_else(|| invalid("CMYK input bytes overflow"))?;
        if pixels.len() != expected {
            return Err(invalid("CMYK input buffer has the wrong size"));
        }
        let profile = self
            .metadata
            .and_then(|metadata| metadata.icc_profile)
            .ok_or_else(|| invalid("CMYK requires an ICC profile"))?;
        let fallible = self
            .limits
            .as_ref()
            .is_some_and(|limits| limits.fallible_alloc());
        let wide = self.layout.is_16bit();
        let bytes_per_sample = if wide { 2 } else { 1 };
        let bits = self.bits_per_sample.unwrap_or(if wide { 16 } else { 8 });
        let maximum = ((1_u64 << bits) - 1) as f32;
        let external = self.extra_channels.iter().find(|extra| {
            extra.info.ec_type == crate::headers::extra_channels::ExtraChannelType::Black
        });
        if external.is_some_and(|extra| extra.info.dim_shift != 0) {
            return Err(invalid(
                "lossy CMYK conversion does not support Black dim_shift > 0",
            ));
        }
        let black = match external {
            None => Black::Interleaved,
            Some(extra) => match extra.data {
                ExtraChannelBuf::U8(values) => Black::U8(
                    values,
                    ((1_u64 << extra.info.bit_depth.bits_per_sample) - 1) as f32,
                ),
                ExtraChannelBuf::U16(values) => Black::U16(
                    values,
                    ((1_u64 << extra.info.bit_depth.bits_per_sample) - 1) as f32,
                ),
                ExtraChannelBuf::F32(_) => {
                    return Err(invalid("CMYK conversion needs integer Black samples"));
                }
            },
        };
        let mut k_guard = BudgetGuard::inert();
        let (mut k8, mut k16) = (None, None);
        if self.layout.is_cmyk() {
            k_guard = MemoryBudget::reserve_opt(Some(budget), (count * bytes_per_sample) as u64)
                .map_err(EncodeError::from)?;
            if wide {
                let mut values = crate::budget::vec_with_capacity_fallible(fallible, count)
                    .map_err(EncodeError::from)?;
                for row in pixels.chunks_exact(width * 8) {
                    crate::error::check_stop(self.stop)?;
                    for pixel in row.as_chunks::<8>().0 {
                        values.push(u16::from_ne_bytes([pixel[6], pixel[7]]));
                    }
                }
                k16 = Some(values);
            } else {
                let mut values = crate::budget::vec_with_capacity_fallible(fallible, count)
                    .map_err(EncodeError::from)?;
                for row in pixels.chunks_exact(width * 4) {
                    crate::error::check_stop(self.stop)?;
                    for pixel in row.as_chunks::<4>().0 {
                        values.push(pixel[3]);
                    }
                }
                k8 = Some(values);
            }
        }
        let channels = self.layout.bytes_per_pixel() / bytes_per_sample;
        let (linear_rgb, guard) = to_linear_srgb(
            InkRows {
                pixels,
                width,
                rows,
                stride: width * self.layout.bytes_per_pixel(),
                channels,
                bytes_per_sample,
                maximum,
                order: if matches!(self.layout, PixelLayout::Bgr8 | PixelLayout::Bgra8) {
                    [2, 1, 0]
                } else {
                    [0, 1, 2]
                },
                black,
            },
            profile,
            budget,
            fallible,
            self.stop,
        )?;
        let alpha = if self.layout.has_alpha() {
            Some(if wide {
                super::ingest::extract_alpha_u16(pixels, 4, 3, maximum, self.stop)
            } else {
                super::ingest::extract_alpha(pixels, 4, 3, self.stop)
            })
        } else {
            None
        };
        crate::error::check_stop(self.stop)?;
        Ok(super::LossyPixels {
            linear_rgb,
            alpha,
            bit_depth_16: wide,
            synthesised_black_u8: k8,
            synthesised_black_u16: k16,
            cmyk_guard: [guard, k_guard],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};
    struct CancelAfter(usize, AtomicUsize);
    impl enough::Stop for CancelAfter {
        fn check(&self) -> Result<(), enough::StopReason> {
            if self.1.fetch_add(1, Ordering::Relaxed) >= self.0 {
                Err(enough::StopReason::Cancelled)
            } else {
                Ok(())
            }
        }
    }
    #[test]
    fn cmyk_rows_ignore_padding_and_release_output_and_scratch_on_refusal() {
        let profile: Vec<u8> = include_str!("../../tests/fixtures/ps_cmyk.hex")
            .split_whitespace()
            .flat_map(|line| {
                (0..line.len())
                    .step_by(2)
                    .map(move |i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
            })
            .collect();
        let packed = [64_u8, 128, 192, 224].repeat(4);
        let mut padded = vec![0xa5; 24];
        padded[..8].copy_from_slice(&packed[..8]);
        padded[12..20].copy_from_slice(&packed[8..]);
        fn input(pixels: &[u8], stride: usize) -> InkRows<'_> {
            InkRows {
                pixels,
                width: 2,
                rows: 2,
                stride,
                channels: 4,
                bytes_per_sample: 1,
                maximum: 255.0,
                order: [0, 1, 2],
                black: Black::Interleaved,
            }
        }
        for fallible in [false, true] {
            let budget = MemoryBudget::new(80);
            let (expected, guard) =
                to_linear_srgb(input(&packed, 8), &profile, &budget, fallible, None).unwrap();
            assert_eq!(budget.used(), 48);
            drop(guard);
            assert_eq!(budget.used(), 0);
            let (actual, guard) =
                to_linear_srgb(input(&padded, 12), &profile, &budget, fallible, None).unwrap();
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            drop(guard);
            assert_eq!(budget.used(), 0);
            for cap in [47_u64, 79] {
                let tight = MemoryBudget::new(cap);
                let error = to_linear_srgb(input(&padded, 12), &profile, &tight, fallible, None)
                    .expect_err("output/scratch budget refusal");
                assert!(error.to_string().contains("memory budget"), "{error}");
                assert_eq!(tight.used(), 0);
            }
            for allowed in [0, 1, 2] {
                let stop = CancelAfter(allowed, AtomicUsize::new(0));
                let error =
                    to_linear_srgb(input(&padded, 12), &profile, &budget, fallible, Some(&stop))
                        .expect_err("cancellation before completion");
                assert!(matches!(error, EncodeError::Cancelled));
                assert_eq!(stop.1.load(Ordering::Relaxed), allowed + 1);
                assert_eq!(budget.used(), 0);
            }
        }
    }
    #[test]
    fn cmyk_source_selector_keeps_rgb_transfer_and_explicit_peak_precedence() {
        use crate::{
            ColorEncoding, ImageMetadata, LossyConfig, PixelLayout, TransferFunction,
            api::ExtraChannel,
        };
        let config = LossyConfig::new(1.0);
        let mut selector = ColorEncoding::srgb();
        selector.want_icc = true;
        selector.transfer_function = TransferFunction::Pq;
        let k = [224_u8];
        let extras = [ExtraChannel::black(&k)];
        let mut normal_color = selector.clone();
        normal_color.want_icc = false;
        let normal = config
            .encode_request(1, 1, PixelLayout::Rgb8)
            .with_color_encoding(normal_color);
        assert!(normal.lossy_source_color_encoding().is_some());
        assert_eq!(normal.resolved_intensity_target(), 10000.0);
        for external in [false, true] {
            let metadata = ImageMetadata::new().with_intensity_target(1000.0);
            let request = config
                .encode_request(
                    1,
                    1,
                    if external {
                        PixelLayout::Rgb8
                    } else {
                        PixelLayout::Cmyk8
                    },
                )
                .with_color_encoding(selector.clone())
                .with_extra_channels(if external { &extras } else { &[] });
            assert!(request.lossy_source_color_encoding().is_none());
            assert_eq!(request.resolved_intensity_target(), 255.0);
            let request = request.with_metadata(&metadata);
            assert_eq!(request.resolved_intensity_target(), 1000.0);
            assert_eq!(
                request
                    .with_intensity_target(2000.0)
                    .resolved_intensity_target(),
                2000.0
            );
        }
    }

    #[test]
    fn cmyk_preparation_releases_black_and_color_reservations_on_row_cancellation() {
        use crate::{ImageMetadata, LossyConfig, PixelLayout};
        let profile: Vec<u8> = include_str!("../../tests/fixtures/ps_cmyk.hex")
            .split_whitespace()
            .flat_map(|line| {
                (0..line.len())
                    .step_by(2)
                    .map(move |i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
            })
            .collect();
        let metadata = ImageMetadata::new().with_icc_profile(&profile);
        let config = LossyConfig::new(0.01);
        for wide in [false, true] {
            for fallible in [false, true] {
                let pixels: Vec<u8> = [64_u8, 128, 192, 224]
                    .into_iter()
                    .flat_map(|v| {
                        if wide {
                            (u16::from(v) * 257).to_ne_bytes().to_vec()
                        } else {
                            vec![v]
                        }
                    })
                    .collect::<Vec<_>>()
                    .repeat(4);
                let limits = crate::Limits::default().with_fallible_alloc(fallible);
                let layout = if wide {
                    PixelLayout::Cmyk16
                } else {
                    PixelLayout::Cmyk8
                };
                for allowed in 0..=5 {
                    let stop = CancelAfter(allowed, AtomicUsize::new(0));
                    let budget = MemoryBudget::unbounded();
                    let request = config
                        .encode_request(2, 2, layout)
                        .with_metadata(&metadata)
                        .with_limits(&limits)
                        .with_stop(&stop);
                    let error = request
                        .prepare_cmyk_pixels(&pixels, 2, 2, &budget)
                        .err()
                        .expect("preparation cancellation");
                    assert!(matches!(error, EncodeError::Cancelled), "{error}");
                    assert_eq!(stop.1.load(Ordering::Relaxed), allowed + 1);
                    assert_eq!(budget.used(), 0);
                }
                let black_bytes = if wide { 8_u64 } else { 4 };
                for cap in [black_bytes - 1, black_bytes + 47, black_bytes + 79] {
                    let budget = MemoryBudget::new(cap);
                    let request = config
                        .encode_request(2, 2, layout)
                        .with_metadata(&metadata)
                        .with_limits(&limits);
                    let error = request
                        .prepare_cmyk_pixels(&pixels, 2, 2, &budget)
                        .err()
                        .expect("Black/color/scratch refusal");
                    assert!(error.to_string().contains("memory budget"), "{error}");
                    assert_eq!(budget.used(), 0);
                }
            }
        }
    }
}
