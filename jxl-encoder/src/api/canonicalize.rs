//! Exact, two-pass input reduction through zenpixels-convert.

use super::*;
use alloc::sync::Arc;
use zenpixels_convert::{
    AlphaMode, ChannelOrder, ColorOrigin, ColorSpace, ImageSpec, PixelBuf, PixelView, SampleLayout,
    SampleType, Transfer, icc::IccIdentification, icc_profiles::SynthesizedIcc, scan_facts,
};

type Prepared = (PixelBuf, PixelLayout, crate::budget::BudgetGuard);

const LAYOUTS: &[(PixelLayout, SampleLayout)] = &[
    (PixelLayout::Rgb8, SampleLayout::RGB8),
    (PixelLayout::Rgba8, SampleLayout::RGBA8),
    (PixelLayout::Bgra8, SampleLayout::BGRA8),
    (PixelLayout::Gray8, SampleLayout::GRAY8),
    (PixelLayout::GrayAlpha8, SampleLayout::GRAYA8),
    (PixelLayout::Rgb16, SampleLayout::RGB16),
    (PixelLayout::Rgba16, SampleLayout::RGBA16),
    (PixelLayout::Gray16, SampleLayout::GRAY16),
    (PixelLayout::GrayAlpha16, SampleLayout::GRAYA16),
    (PixelLayout::RgbLinearF32, SampleLayout::RGBF32),
    (PixelLayout::RgbaLinearF32, SampleLayout::RGBAF32),
    (PixelLayout::GrayLinearF32, SampleLayout::GRAYF32),
    (PixelLayout::GrayAlphaLinearF32, SampleLayout::GRAYAF32),
];

fn sample_spec(layout: SampleLayout) -> ImageSpec {
    let color = if layout.order.is_gray_family() {
        ColorSpace::GRAY_SRGB
    } else {
        ColorSpace::SRGB
    };
    let transfer = if layout.sample == SampleType::F32 {
        Transfer::Linear
    } else {
        Transfer::Srgb
    };
    let alpha = if layout.order.has_alpha_slot() {
        AlphaMode::Straight
    } else {
        AlphaMode::None
    };
    ImageSpec::new(layout, color.with_transfer(transfer).with_alpha(alpha))
}

fn output_icc(icc: Option<&[u8]>) -> Option<SynthesizedIcc> {
    icc.and_then(IccIdentification::from_icc_substitutable)
        .and_then(|identification| identification.to_cicp())
        .map(SynthesizedIcc::for_gray_cicp)
}

pub(super) fn prepare(
    request: &EncodeRequest<'_>,
    pixels: &[u8],
    budget: &Arc<crate::budget::MemoryBudget>,
) -> Result<Option<Prepared>> {
    if let Some(stop) = request.stop {
        stop.check().map_err(|_| at!(EncodeError::Cancelled))?;
    }
    // The u16 predicate tests full-range bit replication (257 * k).
    // Low-bit 10/12/14-bit samples have a different normalization contract.
    if request.layout.is_16bit() && request.bits_per_sample.is_some_and(|b| b != 16) {
        return Ok(None);
    }
    let Some((_, source)) = LAYOUTS.iter().find(|(layout, _)| *layout == request.layout) else {
        return Ok(None);
    };
    let source = *source;
    let source_spec = sample_spec(source);
    let row_bytes = request.width as usize * source.bytes_per_pixel();
    let stride = request.row_stride.unwrap_or(row_bytes);
    let icc = request.metadata.and_then(|m| m.icc_profile);
    // Cover alignment repair and reduced output within the existing budget.
    let bytes = (row_bytes as u64)
        .checked_mul(u64::from(request.height))
        .and_then(|n| n.checked_add(16))
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(icc.map_or(0, |p| p.len() as u64)))
        .ok_or_else(|| {
            at!(EncodeError::LimitExceeded {
                message: "canonicalization allocation size overflow".into()
            })
        })?;
    let guard = budget
        .reserve(bytes)
        .map_err(|e| at(EncodeError::from(e)))?;
    let aligned;
    let view = match PixelView::new(
        pixels,
        request.width,
        request.height,
        stride,
        source_spec.clone(),
    ) {
        Ok(view) => view,
        Err(_) => {
            // EncodeRequest accepts byte addresses and strides without sample alignment.
            let mut copy = PixelBuf::new(request.width, request.height, source_spec)
                .map_err(canonicalization_error)?;
            let mut out = copy.as_view_mut();
            for y in 0..request.height {
                let start = y as usize * stride;
                out.row_mut(y)
                    .copy_from_slice(&pixels[start..start + row_bytes]);
            }
            aligned = copy;
            aligned.as_view()
        }
    };
    let facts = scan_facts(&view);
    let gray_icc = output_icc(icc);
    let collapse = source.order.is_rgb_family()
        && facts.exact.all_grayscale
        && (icc.is_none()
            || matches!(
                gray_icc,
                Some(SynthesizedIcc::Profile(_) | SynthesizedIcc::NotNeeded)
            ));
    let remove_alpha = source.order.has_alpha_slot() && facts.exact.all_opaque;
    let narrow = source.sample == SampleType::U16 && facts.exact.effective_bits == 8;
    if !collapse && !remove_alpha && !narrow {
        return Ok(None);
    }
    let order = match (collapse, remove_alpha, source.order) {
        (true, _, ChannelOrder::Rgb) | (true, true, _) => ChannelOrder::Gray,
        (true, false, _) => ChannelOrder::GrayAlpha,
        (false, true, ChannelOrder::GrayAlpha) => ChannelOrder::Gray,
        (false, true, _) => ChannelOrder::Rgb,
        _ => source.order,
    };
    let target = SampleLayout::new(
        if narrow {
            SampleType::U8
        } else {
            source.sample
        },
        order,
    );
    let Some((layout, _)) = LAYOUTS.iter().find(|(_, samples)| *samples == target) else {
        return Ok(None);
    };
    let retained_icc = if collapse {
        match gray_icc {
            Some(SynthesizedIcc::Profile(bytes)) => Some(Arc::from(bytes.as_ref())),
            _ => None,
        }
    } else {
        icc.map(Arc::from)
    };
    let mut buffer = PixelBuf::new(request.width, request.height, sample_spec(target))
        .map_err(canonicalization_error)?;
    if let Some(icc) = retained_icc {
        buffer = buffer.with_origin(ColorOrigin::from_icc(icc));
    }
    let sample_bytes = source.sample.byte_size();
    let dst_bytes = target.sample.byte_size();
    let mut out = buffer.as_view_mut();
    for y in 0..request.height {
        for (src, dst) in view
            .row(y)
            .chunks_exact(source.bytes_per_pixel())
            .zip(out.row_mut(y).chunks_exact_mut(target.bytes_per_pixel()))
        {
            for (c, word) in dst.chunks_exact_mut(dst_bytes).enumerate() {
                let src_c = if collapse && c == 1 {
                    source.channels() - 1
                } else if source.order == ChannelOrder::Bgra && order == ChannelOrder::Rgb {
                    2 - c
                } else {
                    c
                };
                let word_src = &src[src_c * sample_bytes..][..sample_bytes];
                if narrow {
                    // The scan proved both bytes equal, on either endian.
                    word[0] = word_src[0];
                } else {
                    word.copy_from_slice(word_src);
                }
            }
        }
    }
    Ok(Some((buffer, *layout, guard)))
}

fn canonicalization_error(
    error: zenpixels_convert::At<zenpixels_convert::BufferError>,
) -> At<EncodeError> {
    at!(EncodeError::InvalidInput {
        message: format!("canonicalization buffer: {error}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reduced(layout: PixelLayout, pixels: &[u8]) -> Option<(PixelLayout, Vec<u8>)> {
        let cfg = LossyConfig::new(1.0);
        let request = cfg.encode_request(2, 1, layout);
        prepare(&request, pixels, &crate::budget::MemoryBudget::unbounded())
            .unwrap()
            .map(|(b, l, _)| (l, b.as_view().as_strided_bytes().to_vec()))
    }

    #[test]
    fn exact_alpha_and_chroma_predicates() {
        assert_eq!(
            reduced(PixelLayout::Rgba8, &[1, 1, 1, 255, 2, 2, 2, 255]),
            Some((PixelLayout::Gray8, vec![1, 2]))
        );
        assert_eq!(
            reduced(PixelLayout::Rgba8, &[1, 1, 1, 0, 2, 2, 2, 0]),
            Some((PixelLayout::GrayAlpha8, vec![1, 0, 2, 0]))
        );
        assert_eq!(
            reduced(PixelLayout::Rgba8, &[1, 1, 2, 254, 2, 2, 2, 255]),
            None
        );
        assert_eq!(
            reduced(PixelLayout::Bgra8, &[1, 2, 3, 255, 4, 5, 6, 255]),
            Some((PixelLayout::Rgb8, vec![3, 2, 1, 6, 5, 4]))
        );
    }

    #[test]
    fn full_depth_only_and_unaligned_strided_input() {
        let cfg = LossyConfig::new(1.0);
        // Deliberately odd row stride and offset; padding must not affect
        // opacity, chroma or bit-replication decisions.
        let mut bytes = [0x85; 19];
        for (y, value) in [257_u16, 514].into_iter().enumerate() {
            let start = 1 + y * 9;
            for (c, v) in [value, value, value, 65535].into_iter().enumerate() {
                bytes[start + c * 2..start + c * 2 + 2].copy_from_slice(&v.to_ne_bytes());
            }
        }
        let request = cfg
            .encode_request(1, 2, PixelLayout::Rgba16)
            .with_row_stride(9);
        let (buffer, layout, _) = prepare(
            &request,
            &bytes[1..],
            &crate::budget::MemoryBudget::unbounded(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(layout, PixelLayout::Gray8);
        assert_eq!(buffer.as_view().row(0), &[1]);
        assert_eq!(buffer.as_view().row(1), &[2]);
        let request = request.with_bits_per_sample(12);
        assert!(
            prepare(
                &request,
                &bytes[1..],
                &crate::budget::MemoryBudget::unbounded()
            )
            .unwrap()
            .is_none()
        );
        let data: Vec<u8> = [257_u16, 258]
            .into_iter()
            .flat_map(u16::to_ne_bytes)
            .collect();
        assert!(reduced(PixelLayout::Gray16, &data).is_none());
    }

    #[test]
    fn unknown_rgb_icc_preserves_chroma_and_metadata() {
        let cfg = LossyConfig::new(1.0);
        let profile = [42_u8; 128];
        let metadata = ImageMetadata::default().with_icc_profile(&profile);
        let request = cfg
            .encode_request(2, 1, PixelLayout::Rgba8)
            .with_metadata(&metadata);
        let (buffer, layout, _) = prepare(
            &request,
            &[1, 1, 1, 255, 2, 2, 2, 255],
            &crate::budget::MemoryBudget::unbounded(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(layout, PixelLayout::Rgb8);
        assert_eq!(buffer.origin().icc.as_deref(), Some(profile.as_slice()));
    }

    #[test]
    fn streaming_rejects_two_pass_canonicalization_before_allocation() {
        let result = LossyConfig::new(1.0).with_canonicalize_input(true).encoder(
            511,
            259,
            PixelLayout::Rgba8,
        );
        assert!(
            matches!(result, Err(e) if matches!(e.error(), EncodeError::InvalidConfig { message } if message.contains("two passes")))
        );
    }

    #[test]
    fn canonicalization_respects_budget_and_cancellation() {
        let cfg = LossyConfig::new(1.0);
        let request = cfg.encode_request(2, 1, PixelLayout::Rgba8);
        assert!(prepare(&request, &[255; 8], &crate::budget::MemoryBudget::new(1)).is_err());
        struct Cancel;
        impl Stop for Cancel {
            fn check(&self) -> core::result::Result<(), enough::StopReason> {
                Err(enough::StopReason::Cancelled)
            }
        }
        let request = request.with_stop(&Cancel);
        assert!(
            matches!(prepare(&request, &[255;8], &crate::budget::MemoryBudget::unbounded()), Err(e) if matches!(e.error(), EncodeError::Cancelled))
        );
    }

    // The caller chooses corpus coverage explicitly (nightly + just recipe).
    // No runtime skips and no synthetic replacement for a missing photograph.
    #[cfg(feature = "corpus-tests")]
    #[test]
    fn canonicalization_corpus_roundtrip() {
        let source = image::open(
            crate::test_helpers::corpus_dir().join("CID22/CID22-512/validation/1418519.png"),
        )
        .unwrap()
        .to_rgb8();
        for (w, h) in [(63, 47), (511, 259)] {
            let rgb = image::imageops::crop_imm(&source, 0, 0, w, h)
                .to_image()
                .into_raw();
            for gray in [false, true] {
                for alpha in [0_u8, 255] {
                    let rgba: Vec<u8> = rgb
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .flat_map(|p| {
                            [
                                p[0],
                                if gray { p[0] } else { p[1] },
                                if gray { p[0] } else { p[2] },
                                alpha,
                            ]
                        })
                        .collect();
                    let (manual, manual_layout) = if gray {
                        (
                            rgba.as_chunks::<4>()
                                .0
                                .iter()
                                .flat_map(|p| {
                                    if alpha == 255 {
                                        vec![p[0]]
                                    } else {
                                        vec![p[0], p[3]]
                                    }
                                })
                                .collect::<Vec<_>>(),
                            if alpha == 255 {
                                PixelLayout::Gray8
                            } else {
                                PixelLayout::GrayAlpha8
                            },
                        )
                    } else if alpha == 255 {
                        (rgb.clone(), PixelLayout::Rgb8)
                    } else {
                        (rgba.clone(), PixelLayout::Rgba8)
                    };
                    let cfg = LossyConfig::new(1.0).with_effort(3);
                    let expected = cfg.encode(&manual, w, h, manual_layout).unwrap();
                    for depth16 in [false, true] {
                        let input = if depth16 {
                            rgba.iter()
                                .flat_map(|v| (u16::from(*v) * 257).to_ne_bytes())
                                .collect()
                        } else {
                            rgba.clone()
                        };
                        let layout = if depth16 {
                            PixelLayout::Rgba16
                        } else {
                            PixelLayout::Rgba8
                        };
                        let actual = cfg
                            .clone()
                            .with_canonicalize_input(true)
                            .encode(&input, w, h, layout)
                            .unwrap();
                        assert_eq!(
                            actual, expected,
                            "{w}x{h} gray={gray} alpha={alpha} depth16={depth16}"
                        );
                        for decode in [
                            crate::test_helpers::decode_with_jxl_rs,
                            crate::test_helpers::decode_with_djxl,
                        ] {
                            let decoded = decode(&actual).unwrap();
                            assert_eq!((decoded.width, decoded.height), (w as usize, h as usize));
                            assert!(decoded.pixels.iter().all(|v| v.is_finite()));
                        }
                    }
                }
            }
        }
    }
}
