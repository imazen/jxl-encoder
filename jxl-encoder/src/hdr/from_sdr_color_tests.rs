//! Row, admission and alpha controls independent of gain quantization.
use super::*;
fn rgba(values: [f32; 4]) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_ne_bytes).collect()
}

#[test]
fn color_prepare_strided_rows_match_packed_without_reading_padding() {
    for layout in [
        HdrPixelLayout::Rgb8,
        HdrPixelLayout::Rgba8,
        HdrPixelLayout::RgbaF32,
    ] {
        for transfer in [
            TransferFunction::Linear,
            TransferFunction::Srgb,
            TransferFunction::Pq,
            TransferFunction::Hlg(zenpixels_convert::HlgMode::Scene),
        ] {
            let (width, rows) = (9_u32, 2_u32);
            let row = width as usize * layout.bytes_per_pixel();
            let pixel = match layout {
                HdrPixelLayout::Rgb8 => vec![51, 102, 153],
                HdrPixelLayout::Rgba8 => vec![51, 102, 153, 128],
                HdrPixelLayout::RgbaF32 => rgba([51.0 / 255.0, 102.0 / 255.0, 153.0 / 255.0, 0.5]),
            };
            let packed = pixel.repeat((width * rows) as usize);
            let stride = row + 5;
            let mut padded = vec![0xff; stride * rows as usize];
            for (source, dest) in packed
                .chunks_exact(row)
                .zip(padded.chunks_exact_mut(stride))
            {
                dest[..row].copy_from_slice(source);
            }
            let color = HdrColorEncoding {
                primaries: ColorPrimaries::Bt709,
                transfer,
            };
            let contiguous = linear_rgba(
                &packed,
                width,
                rows,
                row,
                layout,
                color,
                GainSpace {
                    primaries: ColorPrimaries::Bt2020,
                    white_nits: 255.0,
                },
            )
            .unwrap();
            let strided = linear_rgba(
                &padded,
                width,
                rows,
                stride,
                layout,
                color,
                GainSpace {
                    primaries: ColorPrimaries::Bt2020,
                    white_nits: 255.0,
                },
            )
            .unwrap();
            assert_eq!(strided, contiguous, "{layout:?}/{transfer:?}");
            assert_eq!(contiguous.len(), width as usize * rows as usize * 16);
            assert!(
                contiguous
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|v| f32::from_le_bytes(*v).is_finite())
            );
        }
    }
}

#[test]
fn color_prepare_checks_geometry_samples_and_conversion() {
    let color = HdrColorEncoding::linear(ColorPrimaries::Bt709);
    for (width, rows, stride, pixels) in [
        (0, 1, 4, vec![0; 4]),
        (1, 0, 4, vec![0; 4]),
        (1, 3, usize::MAX, vec![0; 4]),
        (1, 1, 3, vec![0; 4]),
        (1, 1, 4, vec![0; 3]),
        (u32::MAX, 1, usize::MAX, vec![]),
    ] {
        assert!(
            linear_rgba(
                &pixels,
                width,
                rows,
                stride,
                HdrPixelLayout::Rgba8,
                color,
                GainSpace {
                    primaries: ColorPrimaries::Bt709,
                    white_nits: 255.0
                }
            )
            .is_err()
        );
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
        assert!(
            linear_rgba(
                &rgba([value, 0.2, 0.1, 1.0]),
                1,
                1,
                16,
                HdrPixelLayout::RgbaF32,
                color,
                GainSpace {
                    primaries: ColorPrimaries::Bt709,
                    white_nits: 255.0
                }
            )
            .is_err()
        );
    }
    assert!(
        linear_rgba(
            &rgba([0.5, 0.2, 0.1, f32::NAN]),
            1,
            1,
            16,
            HdrPixelLayout::RgbaF32,
            color,
            GainSpace {
                primaries: ColorPrimaries::Bt709,
                white_nits: 255.0
            }
        )
        .is_err()
    );
    assert!(
        linear_rgba(
            &rgba([0.5, 0.2, 0.1, 1.0]),
            1,
            1,
            16,
            HdrPixelLayout::RgbaF32,
            color,
            GainSpace {
                primaries: ColorPrimaries::Unknown,
                white_nits: 255.0
            }
        )
        .is_err()
    );
    let wide = HdrColorEncoding::linear(ColorPrimaries::Bt2020);
    assert!(
        linear_rgba(
            &rgba([f32::MAX, 0.0, 0.0, 1.0]),
            1,
            1,
            16,
            HdrPixelLayout::RgbaF32,
            wide,
            GainSpace {
                primaries: ColorPrimaries::Bt709,
                white_nits: 255.0
            }
        )
        .is_err()
    );
}

#[test]
fn color_prepare_matches_native_linear_gamut_vectors() {
    let mut observed = std::collections::BTreeSet::new();
    for line in include_str!("../../tests/fixtures/gain_map_matrices_v012.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter(|line| {
            let input: Vec<f32> = line
                .split('\t')
                .skip(2)
                .take(3)
                .map(|value| value.parse().unwrap())
                .collect();
            [[0.05, 0.05, 0.05], [0.05, 0.03, 0.02], [0.5, 0.2, 0.1]]
                .iter()
                .any(|selected| input == *selected)
        })
    {
        let mut fields = line.split('\t');
        let primaries = |name| match name {
            "srgb" => ColorPrimaries::Bt709,
            "p3" => ColorPrimaries::DisplayP3,
            "bt2020" => ColorPrimaries::Bt2020,
            _ => panic!("unknown native gamut"),
        };
        let from_name = fields.next().unwrap();
        let to_name = fields.next().unwrap();
        let from = primaries(from_name);
        let to = primaries(to_name);
        observed.insert((from_name, to_name));
        let values: Vec<f32> = fields.map(|v| v.parse().unwrap()).collect();
        let result = linear_rgba(
            &rgba([values[0], values[1], values[2], 1.0]),
            1,
            1,
            16,
            HdrPixelLayout::RgbaF32,
            HdrColorEncoding::linear(from),
            GainSpace {
                primaries: to,
                white_nits: 255.0,
            },
        )
        .unwrap();
        for (c, expected) in values[3..].iter().enumerate() {
            let actual = f32::from_le_bytes(result[c * 4..c * 4 + 4].try_into().unwrap());
            assert!(
                (actual - expected).abs() < 2e-6,
                "{line}: channel{c} actual{actual}"
            );
        }
    }
    assert_eq!(observed.len(), 9, "every ordered gamut pair must run");
}

#[test]
fn color_prepare_requires_known_descriptions_and_matching_source_override() {
    for primaries in [ColorPrimaries::Unknown, ColorPrimaries::AdobeRgb] {
        assert!(encoding(HdrColorEncoding::linear(primaries)).is_err());
    }
    for transfer in [
        TransferFunction::Unknown,
        TransferFunction::Bt709,
        TransferFunction::Gamma22,
    ] {
        assert!(
            encoding(HdrColorEncoding {
                primaries: ColorPrimaries::Bt709,
                transfer
            })
            .is_err()
        );
    }
    let linear = ColorEncoding::linear_srgb();
    assert!(source_override(None, linear.clone()).is_err());
    assert!(source_override(None, ColorEncoding::srgb()).is_ok());
    use crate::headers::color_encoding::{
        CIExy, ColorSpace, CustomPrimaries, Primaries, RenderingIntent, TransferFunction as JxlTf,
        WhitePoint,
    };
    for field in 0..8 {
        let mut supplied = linear.clone();
        match field {
            0 => supplied.color_space = ColorSpace::Gray,
            1 => supplied.want_icc = true,
            2 => supplied.white_point = WhitePoint::Dci,
            3 => supplied.primaries = Primaries::P3,
            4 => supplied.transfer_function = JxlTf::Srgb,
            5 => supplied.gamma = Some(1.0),
            6 => supplied.custom_white_point = Some(CIExy::new(0.3127, 0.3290)),
            7 => {
                supplied.custom_primaries = Some(CustomPrimaries {
                    red: CIExy::new(0.64, 0.33),
                    green: CIExy::new(0.3, 0.6),
                    blue: CIExy::new(0.15, 0.06),
                })
            }
            _ => unreachable!(),
        }
        assert!(
            source_override(Some(supplied), linear.clone()).is_err(),
            "override field{field}"
        );
    }
    for intent in [
        RenderingIntent::Perceptual,
        RenderingIntent::Relative,
        RenderingIntent::Saturation,
        RenderingIntent::Absolute,
    ] {
        let mut supplied = linear.clone();
        supplied.rendering_intent = intent;
        assert!(source_override(Some(supplied), linear.clone()).is_ok());
    }
}

#[test]
fn color_prepare_alpha_is_inherited_exactly() {
    for alpha in [0.0_f32, -0.0, 0.25, 0.5, 1.0] {
        let row: Vec<u8> = [0.05_f32, 0.03, 0.02, alpha]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        matching_alpha_row(&row, &row).unwrap();
        let other: Vec<u8> = [0.05_f32, 0.03, 0.02, if alpha == 0.5 { 0.25 } else { 0.5 }]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert!(matching_alpha_row(&row, &other).is_err());
    }
    for alpha in [f32::NAN, f32::INFINITY, -1.0, 2.0] {
        let row: Vec<u8> = [0.05_f32, 0.03, 0.02, alpha]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert!(matching_alpha_row(&row, &row).is_err());
    }
    assert!(matching_alpha_row(&[0; 15], &[0; 15]).is_err());
    assert!(matching_alpha_row(&[0; 16], &[0; 15]).is_err());
}

#[test]
fn color_prepare_matches_native_pq_components() {
    for line in include_str!("../../tests/fixtures/gain_map_transfer_v012.tsv")
        .lines()
        .filter(|line| line.starts_with("pq\t"))
    {
        let values: Vec<f32> = line
            .split('\t')
            .skip(1)
            .map(|v| v.parse().unwrap())
            .collect();
        let result = linear_rgba(
            &rgba([values[0]; 4]),
            1,
            1,
            16,
            HdrPixelLayout::RgbaF32,
            HdrColorEncoding {
                primaries: ColorPrimaries::Bt2020,
                transfer: TransferFunction::Pq,
            },
            GainSpace {
                primaries: ColorPrimaries::Bt2020,
                white_nits: 10000.0,
            },
        )
        .unwrap();
        for c in 0..3 {
            let actual = f32::from_le_bytes(result[c * 4..c * 4 + 4].try_into().unwrap());
            assert!((actual - values[1]).abs() < 4e-6, "{line} actual {actual}");
        }
    }
}

#[test]
fn color_prepare_matches_native_hlg_color_components() {
    for line in include_str!("../../tests/fixtures/gain_map_transfer_v012.tsv")
        .lines()
        .filter(|line| line.starts_with("hlg1000-bt2020\t"))
    {
        let values: Vec<f32> = line
            .split('\t')
            .skip(1)
            .map(|v| v.parse().unwrap())
            .collect();
        let result = linear_rgba(
            &rgba([values[0], values[1], values[2], 1.0]),
            1,
            1,
            16,
            HdrPixelLayout::RgbaF32,
            HdrColorEncoding {
                primaries: ColorPrimaries::Bt2020,
                transfer: TransferFunction::Hlg(zenpixels_convert::HlgMode::Scene),
            },
            GainSpace {
                primaries: ColorPrimaries::Bt2020,
                white_nits: 1000.0,
            },
        )
        .unwrap();
        let actual: [f32; 3] = core::array::from_fn(|c| {
            f32::from_le_bytes(result[c * 4..c * 4 + 4].try_into().unwrap())
        });
        assert!(
            actual
                .iter()
                .zip(&values[3..])
                .all(|(a, b)| (a - b).abs() < 2e-6),
            "{line}; actual {actual:?}"
        );
    }
}

#[test]
fn color_prepare_absolute_transfers_share_the_base_white_units() {
    for line in include_str!("../../tests/fixtures/gain_map_transfer_v012.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let mut fields = line.split('\t');
        let label = fields.next().unwrap();
        let values: Vec<f32> = fields.map(|v| v.parse().unwrap()).collect();
        let (transfer, peak, input, expected, tolerance) = match label {
            "pq" => (
                TransferFunction::Pq,
                10000.0,
                [values[0]; 3],
                [values[1]; 3],
                4e-6_f32,
            ),
            "hlg1000-bt2020" => (
                TransferFunction::Hlg(zenpixels_convert::HlgMode::Scene),
                1000.0,
                [values[0], values[1], values[2]],
                [values[3], values[4], values[5]],
                2e-6,
            ),
            _ => panic!("unknown native transfer {label}"),
        };
        for white_nits in [255.0_f32, 1000.0, 10000.0] {
            let result = linear_rgba(
                &rgba([input[0], input[1], input[2], 1.0]),
                1,
                1,
                16,
                HdrPixelLayout::RgbaF32,
                HdrColorEncoding {
                    primaries: ColorPrimaries::Bt2020,
                    transfer,
                },
                GainSpace {
                    primaries: ColorPrimaries::Bt2020,
                    white_nits,
                },
            )
            .unwrap();
            for (channel, expected) in expected.iter().enumerate() {
                let relative =
                    f32::from_le_bytes(result[channel * 4..channel * 4 + 4].try_into().unwrap());
                // Compare back in the native oracle's display normalization,
                // retaining its original transfer tolerance across white units.
                let native_units = relative * white_nits / peak;
                assert!(
                    (native_units - expected).abs() < tolerance,
                    "{line} white{white_nits} c{channel}: {relative}"
                );
            }
        }
    }
}

#[test]
fn color_prepare_linear_stays_relative_and_validates_the_white() {
    let pixel = rgba([0.1, 0.2, 4.0, 0.5]);
    for white_nits in [255.0_f32, 1000.0, 10000.0] {
        let result = linear_rgba(
            &pixel,
            1,
            1,
            16,
            HdrPixelLayout::RgbaF32,
            HdrColorEncoding::linear(ColorPrimaries::Bt709),
            GainSpace {
                primaries: ColorPrimaries::Bt709,
                white_nits,
            },
        )
        .unwrap();
        for (actual, expected) in result
            .as_chunks::<4>()
            .0
            .iter()
            .zip([0.1_f32, 0.2, 4.0, 0.5])
        {
            assert_eq!(f32::from_le_bytes(*actual).to_bits(), expected.to_bits());
        }
    }
    for white_nits in [0.0_f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            linear_rgba(
                &pixel,
                1,
                1,
                16,
                HdrPixelLayout::RgbaF32,
                HdrColorEncoding::linear(ColorPrimaries::Bt709),
                GainSpace {
                    primaries: ColorPrimaries::Bt709,
                    white_nits
                }
            )
            .is_err()
        );
    }
    for (transfer, expected) in [
        (TransferFunction::Linear, 255.0),
        (TransferFunction::Srgb, 255.0),
        (TransferFunction::Pq, 10000.0),
        (
            TransferFunction::Hlg(zenpixels_convert::HlgMode::Scene),
            1000.0,
        ),
    ] {
        assert_eq!(
            base_white_nits(HdrColorEncoding {
                primaries: ColorPrimaries::Bt709,
                transfer
            }),
            expected
        );
    }
}

#[test]
fn color_prepare_rejects_unknown_descriptions_at_direct_row_entry() {
    for color in [
        HdrColorEncoding {
            primaries: ColorPrimaries::Unknown,
            transfer: TransferFunction::Linear,
        },
        HdrColorEncoding {
            primaries: ColorPrimaries::Bt709,
            transfer: TransferFunction::Unknown,
        },
    ] {
        assert!(
            linear_rgba(
                &rgba([0.1, 0.2, 0.3, 1.0]),
                1,
                1,
                16,
                HdrPixelLayout::RgbaF32,
                color,
                GainSpace {
                    primaries: ColorPrimaries::Bt709,
                    white_nits: 255.0
                }
            )
            .is_err()
        );
    }
}
