//! Exact, two-pass input reduction through zenpixels-convert.

use super::*;
use alloc::sync::Arc;
use zenpixels_convert::{
    ColorContext, PixelBuffer, PixelDescriptor, PixelFormat, PixelSlice, PixelSliceLoadBearingExt,
};

type Prepared = (PixelBuffer, PixelLayout, crate::budget::BudgetGuard);

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
    let format = match request.layout {
        PixelLayout::Rgb8 => PixelFormat::Rgb8,
        PixelLayout::Rgba8 => PixelFormat::Rgba8,
        PixelLayout::Bgra8 => PixelFormat::Bgra8,
        PixelLayout::Gray8 => PixelFormat::Gray8,
        PixelLayout::GrayAlpha8 => PixelFormat::GrayA8,
        PixelLayout::Rgb16 => PixelFormat::Rgb16,
        PixelLayout::Rgba16 => PixelFormat::Rgba16,
        PixelLayout::Gray16 => PixelFormat::Gray16,
        PixelLayout::GrayAlpha16 => PixelFormat::GrayA16,
        PixelLayout::RgbLinearF32 => PixelFormat::RgbF32,
        PixelLayout::RgbaLinearF32 => PixelFormat::RgbaF32,
        PixelLayout::GrayLinearF32 => PixelFormat::GrayF32,
        PixelLayout::GrayAlphaLinearF32 => PixelFormat::GrayAF32,
        // No exact reduction is established for these input contracts.
        _ => return Ok(None),
    };
    let descriptor = PixelDescriptor::from_pixel_format(format);
    let stride = request
        .row_stride
        .unwrap_or_else(|| descriptor.aligned_stride(request.width));
    let icc = request.metadata.and_then(|m| m.icc_profile);
    // Cover both an alignment repair and the reduced output, including
    // allocation alignment slack. The source request was preflighted before
    // this reservation; this guard remains live through the encode.
    let bytes = (descriptor.aligned_stride(request.width) as u64)
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
    let mut view = match PixelSlice::new(pixels, request.width, request.height, stride, descriptor)
    {
        Ok(view) => view,
        Err(_) => {
            // EncodeRequest accepts unaligned byte slices and row strides.
            // Align only this path; tight aligned input remains borrowed.
            let mut copy = PixelBuffer::try_new(request.width, request.height, descriptor)
                .map_err(|e| {
                    at!(EncodeError::InvalidInput {
                        message: format!("canonicalization buffer: {e}")
                    })
                })?;
            let row_bytes = descriptor.aligned_stride(request.width);
            for y in 0..request.height {
                let start = y as usize * stride;
                copy.as_slice_mut()
                    .row_mut(y)
                    .copy_from_slice(&pixels[start..start + row_bytes]);
            }
            aligned = copy;
            aligned.as_slice()
        }
    };
    if let Some(icc) = icc {
        view = view.with_color_context(Arc::new(ColorContext::from_icc(icc)));
    }
    let Some(buffer) = view.try_reduce_to_load_bearing_format() else {
        return Ok(None);
    };
    let layout = match buffer.descriptor().format {
        PixelFormat::Rgb8 => PixelLayout::Rgb8,
        PixelFormat::Rgba8 => PixelLayout::Rgba8,
        PixelFormat::Bgra8 => PixelLayout::Bgra8,
        PixelFormat::Gray8 => PixelLayout::Gray8,
        PixelFormat::GrayA8 => PixelLayout::GrayAlpha8,
        PixelFormat::Rgb16 => PixelLayout::Rgb16,
        PixelFormat::Rgba16 => PixelLayout::Rgba16,
        PixelFormat::Gray16 => PixelLayout::Gray16,
        PixelFormat::GrayA16 => PixelLayout::GrayAlpha16,
        PixelFormat::RgbF32 => PixelLayout::RgbLinearF32,
        PixelFormat::RgbaF32 => PixelLayout::RgbaLinearF32,
        PixelFormat::GrayF32 => PixelLayout::GrayLinearF32,
        PixelFormat::GrayAF32 => PixelLayout::GrayAlphaLinearF32,
        _ => return Ok(None),
    };
    Ok(Some((buffer, layout, guard)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reduced(layout: PixelLayout, pixels: &[u8]) -> Option<(PixelLayout, Vec<u8>)> {
        let cfg = LossyConfig::new(1.0);
        let request = cfg.encode_request(2, 1, layout);
        prepare(&request, pixels, &crate::budget::MemoryBudget::unbounded())
            .unwrap()
            .map(|(b, l, _)| (l, b.as_slice().as_strided_bytes().to_vec()))
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
        assert_eq!(buffer.as_slice().row(0), &[1]);
        assert_eq!(buffer.as_slice().row(1), &[2]);
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
        assert_eq!(
            buffer.as_slice().color_context().unwrap().icc.as_deref(),
            Some(profile.as_slice())
        );
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
                        .chunks_exact(3)
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
                            rgba.chunks_exact(4)
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
