// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Profile conversion for the optional RGB8 coefficient path.
use crate::EncodeError;
use crate::budget::{BudgetGuard, MemoryBudget};
use alloc::{sync::Arc, vec::Vec};
use zenpixels_convert::{
    ColorPrimaries, ColorProfileSource, PixelFormat, PluggableCms, TransferFunction,
};

pub(super) struct RgbRows<'a> {
    pub pixels: &'a [f32],
    pub width: usize,
    pub rows: usize,
    pub stride: usize,
    pub source_matrix: Option<[[f32; 3]; 3]>,
}

pub(super) fn rgb_to_icc(
    input: RgbRows<'_>,
    profile: &[u8],
    budget: &Arc<MemoryBudget>,
    fallible: bool,
    stop: Option<&dyn enough::Stop>,
) -> Result<(Vec<u8>, BudgetGuard), EncodeError> {
    let invalid = |message: &str| EncodeError::InvalidInput {
        message: message.into(),
    };
    let row_len = input
        .width
        .checked_mul(3)
        .ok_or_else(|| invalid("ICC row size overflows"))?;
    let length = row_len
        .checked_mul(input.rows)
        .ok_or_else(|| invalid("ICC image size overflows"))?;
    let extent = input
        .rows
        .checked_sub(1)
        .and_then(|rows| rows.checked_mul(input.stride))
        .and_then(|offset| offset.checked_add(row_len))
        .ok_or_else(|| invalid("ICC input extent overflows"))?;
    if row_len == 0 || input.stride < row_len || input.pixels.len() < extent {
        return Err(invalid("ICC input rows are incomplete"));
    }
    let width = u32::try_from(input.width).map_err(|_| invalid("ICC row width exceeds u32"))?;
    let mut transform = zenpixels_convert::MoxCms
        .build_source_transform(
            ColorProfileSource::PrimariesTransferPair {
                primaries: ColorPrimaries::Bt709,
                transfer: TransferFunction::Linear,
            },
            ColorProfileSource::Icc(profile),
            PixelFormat::RgbF32,
            PixelFormat::RgbF32,
            &zenpixels_convert::policy::ConvertOptions::permissive().with_clip_out_of_gamut(false),
        )
        .ok_or_else(|| EncodeError::InvalidConfig {
            message: "requested source-to-ICC conversion is unavailable".into(),
        })?
        .map_err(|error| EncodeError::InvalidConfig {
            message: format!("source-to-ICC conversion: {error}"),
        })?;
    transform
        .prepare(width)
        .map_err(|error| EncodeError::InvalidConfig {
            message: format!("source-to-ICC conversion: {error}"),
        })?;
    let output_guard =
        MemoryBudget::reserve_opt(Some(budget), length as u64).map_err(EncodeError::from)?;
    let mut output = crate::budget::vec_with_capacity_fallible::<u8>(fallible, length)
        .map_err(EncodeError::from)?;
    output.resize(length, 0);
    let scratch_rows = if input.source_matrix.is_some() { 3 } else { 1 };
    let scratch_bytes = row_len
        .checked_mul(scratch_rows)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| invalid("ICC row scratch size overflows"))?;
    let _scratch_guard =
        MemoryBudget::reserve_opt(Some(budget), scratch_bytes as u64).map_err(EncodeError::from)?;
    let mut converted =
        crate::budget::vec_f32_zeroed_fallible(fallible, row_len).map_err(EncodeError::from)?;
    let mut matrix_rows = if input.source_matrix.is_some() {
        Some((
            crate::budget::vec_f32_zeroed_fallible(fallible, row_len).map_err(EncodeError::from)?,
            crate::budget::vec_f32_zeroed_fallible(fallible, row_len).map_err(EncodeError::from)?,
        ))
    } else {
        None
    };
    for y in 0..input.rows {
        crate::error::check_stop(stop)?;
        let source = &input.pixels[y * input.stride..][..row_len];
        let source = if let (Some(matrix), Some((planar, packed))) =
            (input.source_matrix.as_ref(), matrix_rows.as_mut())
        {
            let (red, gb) = planar.split_at_mut(input.width);
            let (green, blue) = gb.split_at_mut(input.width);
            for (index, pixel) in source.as_chunks::<3>().0.iter().enumerate() {
                red[index] = pixel[0];
                green[index] = pixel[1];
                blue[index] = pixel[2];
            }
            crate::vardct::xyb::apply_matrix_3x3(red, green, blue, matrix);
            for (index, pixel) in packed.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                *pixel = [red[index], green[index], blue[index]];
            }
            packed.as_slice()
        } else {
            source
        };
        if source.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "source-to-ICC conversion needs finite source samples",
            ));
        }
        transform
            .try_transform_row(
                bytemuck::cast_slice(source),
                bytemuck::cast_slice_mut(&mut converted),
                width,
            )
            .map_err(|error| EncodeError::InvalidInput {
                message: format!("source-to-ICC row conversion: {error}"),
            })?;
        for (destination, &sample) in output[y * row_len..][..row_len].iter_mut().zip(&converted) {
            if !sample.is_finite() {
                return Err(invalid(
                    "source-to-ICC conversion produced a non-finite sample",
                ));
            }
            *destination = (sample.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    Ok((output, output_guard))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};
    const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    #[test]
    fn icc_rows_ignore_nan_padding_and_release_scratch() {
        let packed = [0.4, 0.2, 0.1, 0.5, 0.3, 0.15].repeat(2);
        let mut padded = vec![f32::NAN; 18];
        padded[..6].copy_from_slice(&packed[..6]);
        padded[9..15].copy_from_slice(&packed[6..]);
        for matrix in [None, Some(IDENTITY)] {
            let budget = MemoryBudget::new(4096);
            let baseline = rgb_to_icc(
                RgbRows {
                    pixels: &packed,
                    width: 2,
                    rows: 2,
                    stride: 6,
                    source_matrix: matrix,
                },
                zenpixels_convert::icc_profiles::DISPLAY_P3_V4,
                &budget,
                true,
                None,
            )
            .unwrap();
            assert_eq!(
                budget.used(),
                12,
                "only the returned output remains reserved"
            );
            let strided = rgb_to_icc(
                RgbRows {
                    pixels: &padded,
                    width: 2,
                    rows: 2,
                    stride: 9,
                    source_matrix: matrix,
                },
                zenpixels_convert::icc_profiles::DISPLAY_P3_V4,
                &budget,
                true,
                None,
            )
            .unwrap();
            assert_eq!(baseline.0, strided.0);
            drop(baseline);
            drop(strided);
            assert_eq!(budget.used(), 0);
        }
    }

    struct StopAfterFirstRow(AtomicUsize);
    impl enough::Stop for StopAfterFirstRow {
        fn check(&self) -> Result<(), enough::StopReason> {
            if self.0.fetch_add(1, Ordering::Relaxed) == 0 {
                Ok(())
            } else {
                Err(enough::StopReason::Cancelled)
            }
        }
    }

    #[test]
    fn icc_row_errors_release_known_buffer_reservations() {
        let pixels = [0.5; 12];
        for case in [
            "budget",
            "cancel",
            "nonfinite",
            "matrix-overflow",
            "short-row",
        ] {
            let budget = MemoryBudget::new(if case == "budget" { 11 } else { 4096 });
            let stop = StopAfterFirstRow(AtomicUsize::new(0));
            let mut source = pixels;
            if case == "nonfinite" {
                source[6] = f32::NAN;
            }
            let matrix = (case == "matrix-overflow").then_some([[f32::MAX; 3]; 3]);
            let result = rgb_to_icc(
                RgbRows {
                    pixels: &source,
                    width: 2,
                    rows: 2,
                    stride: if case == "short-row" { 5 } else { 6 },
                    source_matrix: matrix,
                },
                zenpixels_convert::icc_profiles::DISPLAY_P3_V4,
                &budget,
                true,
                (case == "cancel").then_some(&stop as &dyn enough::Stop),
            );
            assert!(result.is_err(), "{case}");
            if case == "cancel" {
                assert!(format!("{:?}", result.err().unwrap()).contains("Cancelled"));
            }
            assert_eq!(budget.used(), 0, "{case}");
        }
    }
    #[test]
    fn icc_conversion_preserves_extended_intermediate_primaries() {
        // Native LCMS relative-colorimetric source/P3 control, followed by
        // RGB8 quantization: 160,39,7. Negative sRGB blue is in-gamut in P3.
        let rgb = [0.423791885376, 0.006110637449, -0.005660005845];
        let mut padded = [f32::NAN; 9];
        padded[..3].copy_from_slice(&rgb);
        padded[6..].copy_from_slice(&rgb);
        let budget = MemoryBudget::new(4096);
        let (codes, _guard) = rgb_to_icc(
            RgbRows {
                pixels: &padded,
                width: 1,
                rows: 2,
                stride: 6,
                source_matrix: None,
            },
            zenpixels_convert::icc_profiles::DISPLAY_P3_V4,
            &budget,
            true,
            None,
        )
        .unwrap();
        assert_eq!(codes, [160, 39, 7].repeat(2));
    }

    #[test]
    fn invalid_icc_row_shapes_and_scratch_caps_release_reservations() {
        let source = [0.5_f32; 12];
        for (width, rows, stride, length) in [
            (usize::MAX, 1, 3, 0),
            (1, usize::MAX, 3, 0),
            (1, 3, usize::MAX, 0),
            (1, 0, 3, 0),
            (0, 1, 0, 0),
            (1, 1, 2, 3),
            (1, 1, 3, 2),
            (2, 2, 6, 11),
        ] {
            let budget = MemoryBudget::new(4096);
            let result = rgb_to_icc(
                RgbRows {
                    pixels: &source[..length],
                    width,
                    rows,
                    stride,
                    source_matrix: None,
                },
                zenpixels_convert::icc_profiles::DISPLAY_P3_V4,
                &budget,
                true,
                None,
            );
            assert!(
                result.is_err(),
                "invalid rows {width}/{rows}/{stride}/{length}"
            );
            assert_eq!(budget.used(), 0);
        }
        for matrix in [None, Some(IDENTITY)] {
            // Output reservation fits; the later row scratch reservation does not.
            let budget = MemoryBudget::new(30);
            let result = rgb_to_icc(
                RgbRows {
                    pixels: &source,
                    width: 2,
                    rows: 2,
                    stride: 6,
                    source_matrix: matrix,
                },
                zenpixels_convert::icc_profiles::DISPLAY_P3_V4,
                &budget,
                true,
                None,
            );
            assert!(result.is_err());
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn source_copy_budget_precedes_pixel_preparation() {
        let config = crate::LossyConfig::new(0.01);
        let request = config.encode_request(2, 1, crate::PixelLayout::Rgb8);
        let budget = MemoryBudget::new(1);
        let result = request.subsampled_icc_pixels(
            &config,
            &[],
            zenpixels_convert::icc_profiles::DISPLAY_P3_V4,
            &budget,
        );
        assert!(
            matches!(result, Err(EncodeError::LimitExceeded { .. })),
            "the source reservation must precede the invalid empty buffer"
        );
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn source_wrapper_consumes_both_fallible_allocation_settings() {
        let config = crate::LossyConfig::new(0.01);
        let color = crate::ColorEncoding::from_cicp(11, 8, 0, true).unwrap();
        for fallible in [false, true] {
            let limits = crate::Limits::default().with_fallible_alloc(fallible);
            let request = config
                .encode_request(2, 1, crate::PixelLayout::Rgb8)
                .with_limits(&limits)
                .with_color_encoding(color.clone());
            let budget = MemoryBudget::new(4096);
            let result = request
                .subsampled_icc_pixels(
                    &config,
                    &[128, 64, 32, 128, 64, 32],
                    zenpixels_convert::icc_profiles::DISPLAY_P3_V4,
                    &budget,
                )
                .unwrap();
            assert_eq!(
                result.0,
                [186, 137, 99, 186, 137, 99],
                "native LCMS rounded-profile control"
            );
            assert_eq!(budget.used(), 6);
            drop(result);
            assert_eq!(budget.used(), 0);
            let error = request
                .subsampled_icc_pixels(
                    &config,
                    &[],
                    zenpixels_convert::icc_profiles::DISPLAY_P3_V4,
                    &budget,
                )
                .err()
                .unwrap();
            assert!(matches!(error, EncodeError::InvalidInput { .. }));
            assert_eq!(
                budget.used(),
                0,
                "failed source preparation releases its reservation"
            );
        }
    }

    #[test]
    fn source_wrapper_defensive_errors_precede_cms_work() {
        let config = crate::LossyConfig::new(0.01);
        let profile = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
        let budget = MemoryBudget::new(4096);
        let oversized = config.encode_request(u32::MAX, u32::MAX, crate::PixelLayout::Rgb8);
        assert!(
            format!(
                "{:?}",
                oversized
                    .subsampled_icc_pixels(&config, &[], profile, &budget)
                    .err()
                    .unwrap()
            )
            .contains("ICC source size overflows")
        );
        let mut unknown = crate::ColorEncoding::srgb();
        unknown.transfer_function = crate::TransferFunction::Unknown;
        let point = crate::CIExy { x: 0.3, y: 0.3 };
        let singular = crate::ColorEncoding::with_custom_primaries(crate::CustomPrimaries {
            red: point,
            green: point,
            blue: point,
        });
        for (color, diagnostic) in [
            (unknown, "known transfer function"),
            (singular, "determinant"),
        ] {
            let request = config
                .encode_request(2, 1, crate::PixelLayout::Rgb8)
                .with_color_encoding(color);
            let error = request
                .subsampled_icc_pixels(&config, &[128; 6], profile, &budget)
                .err()
                .unwrap();
            assert!(format!("{error:?}").contains(diagnostic), "{error:?}");
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn finite_source_with_nonfinite_cms_output_is_refused() {
        // A valid LCMS-generated RGB matrix/shaper profile with gamma 1/256.
        // Its inverse curve overflows for this extended finite input.
        let hex: alloc::string::String = include_str!("../../tests/fixtures/icc_extreme_gamma.hex")
            .split_whitespace()
            .collect();
        let profile: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let budget = MemoryBudget::new(4096);
        let result = rgb_to_icc(
            RgbRows {
                pixels: &[2.0; 3],
                width: 1,
                rows: 1,
                stride: 3,
                source_matrix: None,
            },
            &profile,
            &budget,
            true,
            None,
        );
        let Err(error) = result else {
            panic!("CMS overflow must be refused before quantization");
        };
        assert!(
            matches!(&error, EncodeError::InvalidInput { .. })
                && format!("{error:?}").contains("non-finite"),
            "{error:?}"
        );
        assert_eq!(budget.used(), 0);
    }
}
