use super::*;
use jxl_encoder::api::ExtraChannel;
#[test]
fn separated_black_and_alpha_keep_progressive_final_samples() {
    let out = directory();
    let profile: Vec<u8> = include_str!("../fixtures/ps_cmyk.hex")
        .split_whitespace()
        .flat_map(|line| {
            (0..line.len())
                .step_by(2)
                .map(move |i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
        })
        .collect();
    let physical: Vec<[f32; 3]> = include_str!("../fixtures/cmyk_lcms_swatches.hex")
        .lines()
        .map(|line| {
            line.split_whitespace()
                .map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap()))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect();
    let meta = ImageMetadata::new().with_icc_profile(&profile);
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("pinned native observer");
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (w, h) in [(32_usize, 17_usize), (259, 17)] {
            let black: Vec<u8> = (0..w * h)
                .map(|i| SWATCHES[swatch(i % w, i / w, 0, false)][3])
                .collect();
            let black16: Vec<u16> = black.iter().map(|v| u16::from(*v) * 257).collect();
            for layout in [
                PixelLayout::Rgb8,
                PixelLayout::Bgr8,
                PixelLayout::Rgba8,
                PixelLayout::Bgra8,
                PixelLayout::Rgb16,
                PixelLayout::Rgba16,
            ] {
                let wide = layout.is_16bit();
                let extras = [if wide {
                    ExtraChannel::black_u16(&black16)
                } else {
                    ExtraChannel::black(&black)
                }];
                let pixels: Vec<u8> = (0..w * h)
                    .flat_map(|i| {
                        let mut p = SWATCHES[swatch(i % w, i / w, 0, false)][..3].to_vec();
                        if matches!(layout, PixelLayout::Bgr8 | PixelLayout::Bgra8) {
                            p.swap(0, 2);
                        }
                        if layout.has_alpha() {
                            p.push(137);
                        }
                        p.into_iter().flat_map(|v| {
                            if wide {
                                (u16::from(v) * 257).to_ne_bytes().to_vec()
                            } else {
                                vec![v]
                            }
                        })
                    })
                    .collect();
                let mut baseline = None;
                for progressive in [
                    ProgressiveMode::Single,
                    ProgressiveMode::QuantizedAcFullAc,
                    ProgressiveMode::DcVlfLfAc,
                ] {
                    let label = format!("{policy}-{w}x{h}-{layout:?}-{progressive:?}");
                    let config = LossyConfig::new(0.01)
                        .with_effort(3)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_force_strategy(Some(3))
                        .with_progressive(progressive);
                    let req = || {
                        config
                            .encode_request(w as u32, h as u32, layout)
                            .with_metadata(&meta)
                            .with_extra_channels(&extras)
                    };
                    let bytes = req().encode(&pixels).unwrap();
                    schedule(&bytes, 1, progressive);
                    let stride = w * layout.bytes_per_pixel() + 11;
                    let mut padded = vec![0xa5; stride * h];
                    for (to, from) in padded
                        .chunks_exact_mut(stride)
                        .zip(pixels.chunks_exact(w * layout.bytes_per_pixel()))
                    {
                        to[..from.len()].copy_from_slice(from);
                    }
                    assert_eq!(
                        req().with_row_stride(stride).encode(&padded).unwrap(),
                        bytes
                    );
                    let mut writer = Vec::new();
                    req().encode_to(&pixels, &mut writer).unwrap();
                    assert_eq!(writer, bytes);
                    let target = out.join(&label);
                    std::fs::create_dir(&target).unwrap();
                    std::fs::write(target.join("source.bin"), &pixels).unwrap();
                    std::fs::write(target.join("profile.icc"), &profile).unwrap();
                    let actual = render(
                        &target.join("render"),
                        &bytes,
                        (w, h),
                        1,
                        (wide, false),
                        &oracle,
                        &physical,
                    );
                    if let Some(expected) = &baseline {
                        same_final(&actual, expected, &label);
                    } else {
                        baseline = Some(actual);
                    }
                }
            }
        }
    }
}
