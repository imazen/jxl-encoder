//! Final progressive color must preserve the single-pass quantized coefficients.
#![cfg(feature = "__expert")]
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout, ProgressiveMode};
use std::{
    io::Write,
    path::{Path, PathBuf},
};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/pfm.rs"]
mod pfm;

fn directory() -> PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("progressive-ac-contract");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap()
}
fn render(
    out: &Path,
    bytes: &[u8],
    label: &str,
    width: usize,
    height: usize,
    layout: PixelLayout,
) -> [Vec<f32>; 2] {
    let input = out.join(format!("{label}.jxl"));
    let output = out.join(format!("{label}.pfm"));
    std::fs::write(&input, bytes).unwrap();
    let (mut primary, extras) = decoded_extras::decode_current_planes(bytes, label);
    assert_eq!(extras.len(), usize::from(layout.has_alpha()));
    let colors = if layout.is_grayscale() { 1 } else { 3 };
    assert_eq!(primary.len(), width * height * colors);
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(&output)
        .args(["--num_threads=1", "--output_extra_channels"])
        .output()
        .unwrap();
    std::fs::write(out.join(format!("{label}.log")), &result.stderr).unwrap();
    assert!(result.status.success(), "{label}: djxl full rendering");
    let mut reference = pfm::read_pfm_samples(&output, (width, height));
    assert_eq!(reference.len(), width * height * colors);
    for (i, plane) in extras.iter().enumerate() {
        assert_eq!(plane.len(), width * height);
        primary.extend_from_slice(plane);
        let reference_plane = pfm::read_pfm_samples(
            &out.join(format!("{label}.pfm-ec{}.pfm", i + 1)),
            (width, height),
        );
        reference.extend_from_slice(&reference_plane);
    }
    assert!(primary.iter().chain(&reference).all(|v| v.is_finite()));
    [primary, reference]
}
#[test]
fn progressive_real_pixels_keep_single_pass_coefficients() {
    let root = directory();
    let fixture = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    let mut journal = std::fs::File::create(root.join("completed.tsv")).unwrap();
    writeln!(
        journal,
        "width\theight\traw_strategy\tmode\tdecoder\twrong_samples\tfirst"
    )
    .unwrap();
    let mut failures = Vec::new();
    for (width, height) in [(64_usize, 64_usize), (259, 133)] {
        let pixels: Vec<u8> = (0..height)
            .flat_map(|y| {
                let fixture = &fixture;
                (0..width).flat_map(move |x| {
                    fixture
                        .get_pixel(x as u32 % fixture.width(), y as u32 % fixture.height())
                        .0
                })
            })
            .collect();
        for raw in 0..19 {
            let out = root.join(format!("{width}x{height}-raw{raw}"));
            std::fs::create_dir(&out).unwrap();
            std::fs::write(out.join("input.rgb8"), &pixels).unwrap();
            let config = LossyConfig::new(1.0)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(EncoderStrategy::Libjxl)
                .with_force_strategy(Some(raw));
            let baseline = config
                .encode(&pixels, width as u32, height as u32, PixelLayout::Rgb8)
                .unwrap();
            let baseline = render(&out, &baseline, "single", width, height, PixelLayout::Rgb8);
            for mode in [
                ProgressiveMode::QuantizedAcFullAc,
                ProgressiveMode::DcVlfLfAc,
            ] {
                let bytes = config
                    .clone()
                    .with_progressive(mode)
                    .encode(&pixels, width as u32, height as u32, PixelLayout::Rgb8)
                    .unwrap();
                let rendered = render(
                    &out,
                    &bytes,
                    &format!("{mode:?}"),
                    width,
                    height,
                    PixelLayout::Rgb8,
                );
                for (decoder, (actual, expected)) in ["jxl-rs", "djxl"]
                    .into_iter()
                    .zip(rendered.iter().zip(&baseline))
                {
                    let mut wrong = 0;
                    let mut first = None;
                    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
                        if a.to_bits() != b.to_bits() {
                            wrong += 1;
                            if first.is_none() {
                                first = Some((i, *a, *b));
                            }
                        }
                    }
                    writeln!(
                        journal,
                        "{width}\t{height}\t{raw}\t{mode:?}\t{decoder}\t{wrong}\t{first:?}"
                    )
                    .unwrap();
                    journal.flush().unwrap();
                    if wrong != 0 {
                        failures.push(format!(
                            "{width}x{height} raw{raw} {mode:?} {decoder}: {wrong} wrong, {first:?}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "progressive final-color failures {failures:?}; {}",
        root.display()
    );
}

fn encode_entry(
    config: &LossyConfig,
    pixels: &[u8],
    width: usize,
    height: usize,
    layout: PixelLayout,
    entry: &str,
) -> Vec<u8> {
    let request = || config.encode_request(width as u32, height as u32, layout);
    match entry {
        "legacy" => config
            .encode(pixels, width as u32, height as u32, layout)
            .unwrap(),
        "request" => request().encode(pixels).unwrap(),
        "writer" => {
            let mut bytes = Vec::new();
            request().encode_to(pixels, &mut bytes).unwrap();
            bytes
        }
        "strided" => {
            let row = width * layout.bytes_per_pixel();
            let stride = row + 3;
            let mut padded = vec![0xa5; stride * height];
            for (source, dest) in pixels
                .chunks_exact(row)
                .zip(padded.chunks_exact_mut(stride))
            {
                dest[..row].copy_from_slice(source);
            }
            request().with_row_stride(stride).encode(&padded).unwrap()
        }
        "stream" => {
            let row = width * layout.bytes_per_pixel();
            let mut encoder = config.encoder(width as u32, height as u32, layout).unwrap();
            for rows in pixels.chunks(row * 7) {
                encoder.push_rows(rows, (rows.len() / row) as u32).unwrap();
            }
            encoder.finish().unwrap()
        }
        _ => panic!("unknown entry {entry}"),
    }
}

struct PixelCase<'a> {
    pixels: &'a [u8],
    width: usize,
    height: usize,
    layout: PixelLayout,
    entry: &'a str,
}
fn check_config(out: &Path, config: LossyConfig, case: PixelCase<'_>) {
    std::fs::create_dir(out).unwrap();
    std::fs::write(out.join("config.txt"), format!("{config:?}\n")).unwrap();
    let PixelCase {
        pixels,
        width,
        height,
        layout,
        entry,
    } = case;
    std::fs::write(out.join("input.bin"), pixels).unwrap();
    let baseline = encode_entry(&config, pixels, width, height, layout, entry);
    let baseline = render(out, &baseline, "single", width, height, layout);
    let mut failures = Vec::new();
    for mode in [
        ProgressiveMode::QuantizedAcFullAc,
        ProgressiveMode::DcVlfLfAc,
    ] {
        let bytes = encode_entry(
            &config.clone().with_progressive(mode),
            pixels,
            width,
            height,
            layout,
            entry,
        );
        let decoded = render(out, &bytes, &format!("{mode:?}"), width, height, layout);
        for (decoder, (actual, expected)) in ["jxl-rs", "djxl"]
            .into_iter()
            .zip(decoded.iter().zip(&baseline))
        {
            assert_eq!(actual.len(), expected.len());
            let wrong: Vec<_> = actual
                .iter()
                .zip(expected)
                .enumerate()
                .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
                .collect();
            let record = format!("wrong={} first={:?}\n", wrong.len(), wrong.first());
            std::fs::write(out.join(format!("{mode:?}-{decoder}.samples")), &record).unwrap();
            if !wrong.is_empty() {
                failures.push(format!("{mode:?} {decoder} {record}"));
            }
        }
    }
    assert!(failures.is_empty(), "{} {failures:?}", out.display());
}
#[path = "support/encoder_policies.rs"]
mod encoder_policies;

#[test]
fn offered_progressive_policies_and_public_entries_preserve_final_pixels() {
    let root = directory();
    let fixture = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    for (index, (policy, strategy, mode)) in encoder_policies::policies().into_iter().enumerate() {
        for (width, height) in [(64, 64), (259, 133)] {
            let pixels: Vec<u8> = (0..height)
                .flat_map(|y| {
                    let fixture = &fixture;
                    (0..width).flat_map(move |x| {
                        fixture
                            .get_pixel(x as u32 % fixture.width(), y as u32 % fixture.height())
                            .0
                    })
                })
                .collect();
            for entry in ["legacy", "request", "writer", "strided", "stream"] {
                let raw = [3, 4, 10, 11, 16, 17, 18][index % 7];
                let config = LossyConfig::new(1.0)
                    .with_effort(4)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_force_strategy(Some(raw));
                check_config(
                    &root.join(format!("{policy}-{mode:?}-{width}x{height}-{entry}")),
                    config,
                    PixelCase {
                        pixels: &pixels,
                        width,
                        height,
                        layout: PixelLayout::Rgb8,
                        entry,
                    },
                );
            }
        }
    }
}

fn typed_pixels(rgb: &[u8], layout: PixelLayout) -> Vec<u8> {
    let mut out = Vec::new();
    for pixel in rgb.as_chunks::<3>().0 {
        let mut values = if layout.is_grayscale() {
            vec![pixel[0]]
        } else {
            pixel.to_vec()
        };
        if matches!(layout, PixelLayout::Bgr8 | PixelLayout::Bgra8) {
            values.reverse();
        }
        if layout.has_alpha() {
            values.push(pixel[1]);
        }
        for value in values {
            if layout.is_f16() {
                // Exact k/16 binary16 fixture values, as in production_resources.
                const HALF: [u16; 16] = [
                    0, 0x2c00, 0x3000, 0x3200, 0x3400, 0x3500, 0x3600, 0x3700, 0x3800, 0x3880,
                    0x3900, 0x3980, 0x3a00, 0x3a80, 0x3b00, 0x3b80,
                ];
                out.extend_from_slice(&HALF[usize::from(value >> 4)].to_ne_bytes());
            } else if layout.is_f32() {
                out.extend_from_slice(&(f32::from(value) / 255.0).to_ne_bytes());
            } else if layout.is_16bit() {
                out.extend_from_slice(&(u16::from(value) * 257).to_ne_bytes());
            } else {
                out.push(value);
            }
        }
    }
    out
}
fn non_cmyk_lossy_layouts() -> [PixelLayout; 24] {
    [
        PixelLayout::Rgb8,
        PixelLayout::Rgba8,
        PixelLayout::Bgr8,
        PixelLayout::Bgra8,
        PixelLayout::Gray8,
        PixelLayout::GrayAlpha8,
        PixelLayout::Rgb16,
        PixelLayout::Rgba16,
        PixelLayout::Gray16,
        PixelLayout::GrayAlpha16,
        PixelLayout::RgbLinearF32,
        PixelLayout::RgbaLinearF32,
        PixelLayout::GrayLinearF32,
        PixelLayout::GrayAlphaLinearF32,
        PixelLayout::RgbLinearF16,
        PixelLayout::RgbaLinearF16,
        PixelLayout::GrayLinearF16,
        PixelLayout::GrayAlphaLinearF16,
        PixelLayout::RgbPqF32,
        PixelLayout::RgbaPqF32,
        PixelLayout::RgbHlgF32,
        PixelLayout::RgbaHlgF32,
        PixelLayout::RgbBt709F32,
        PixelLayout::RgbaBt709F32,
    ]
}
#[test]
fn progressive_typed_layouts_preserve_final_color_and_alpha() {
    let root = directory();
    let fixture = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    for (index, layout) in non_cmyk_lossy_layouts().into_iter().enumerate() {
        let (width, height) = if index % 2 == 0 { (64, 64) } else { (259, 133) };
        let rgb: Vec<u8> = (0..height)
            .flat_map(|y| {
                let fixture = &fixture;
                (0..width).flat_map(move |x| {
                    fixture
                        .get_pixel(x as u32 % fixture.width(), y as u32 % fixture.height())
                        .0
                })
            })
            .collect();
        let pixels = typed_pixels(&rgb, layout);
        let config = LossyConfig::new(1.0)
            .with_effort(4)
            .with_threads(1)
            .with_strategy(EncoderStrategy::Libjxl)
            .with_force_strategy(Some(3));
        check_config(
            &root.join(format!("{layout:?}-{width}x{height}")),
            config,
            PixelCase {
                pixels: &pixels,
                width,
                height,
                layout,
                entry: "request",
            },
        );
    }
}

fn associated_pixels(pixels: &[u8], layout: PixelLayout) -> Vec<u8> {
    assert!(layout.has_alpha() && !layout.is_f32() && !layout.is_f16());
    let mut out = pixels.to_vec();
    for pixel in out.chunks_exact_mut(layout.bytes_per_pixel()) {
        if layout.is_16bit() {
            let alpha = u32::from(u16::from_ne_bytes(
                pixel[pixel.len() - 2..].try_into().unwrap(),
            ));
            let colors = pixel.len() - 2;
            for channel in pixel[..colors].as_chunks_mut::<2>().0 {
                let value = u32::from(u16::from_ne_bytes(*channel));
                *channel = (((value * alpha + 32767) / 65535) as u16).to_ne_bytes();
            }
        } else {
            let alpha = u32::from(pixel[pixel.len() - 1]);
            let colors = pixel.len() - 1;
            for value in &mut pixel[..colors] {
                *value = ((u32::from(*value) * alpha + 127) / 255) as u8;
            }
        }
    }
    out
}

fn render_animation(
    out: &Path,
    label: &str,
    bytes: &[u8],
    width: usize,
    height: usize,
    layout: PixelLayout,
) -> [Vec<Vec<f32>>; 2] {
    let primary = std::panic::catch_unwind(|| {
        let primary = decoded_extras::decode_current_frames(bytes, label);
        assert_eq!(primary.len(), 2);
        let primary: Vec<_> = primary
            .into_iter()
            .map(|(mut color, extras)| {
                assert_eq!(
                    color.len(),
                    width * height * if layout.is_grayscale() { 1 } else { 3 }
                );
                assert_eq!(extras.len(), usize::from(layout.has_alpha()));
                for plane in extras {
                    assert_eq!(plane.len(), width * height);
                    color.extend(plane);
                }
                assert!(color.iter().all(|v| v.is_finite()));
                color
            })
            .collect();
        primary
    });
    let input = out.join(format!("{label}.jxl"));
    std::fs::write(&input, bytes).unwrap();
    let output = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(out.join(format!("{label}.pfm")))
        .args([
            "--output_frames",
            "--output_extra_channels",
            "--num_threads=1",
        ])
        .output()
        .unwrap();
    std::fs::write(out.join(format!("{label}.log")), &output.stderr).unwrap();
    assert!(output.status.success(), "{label}: animation djxl rendering");
    let primary = primary.expect("primary animation decoder rendering");
    let mut paths: Vec<_> = std::fs::read_dir(out)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_str().unwrap();
            name.starts_with(label) && name.ends_with(".pfm") && !name.contains("-ec")
        })
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 2);
    let native = paths
        .into_iter()
        .map(|p| {
            let mut samples = pfm::read_pfm_samples(&p, (width, height));
            assert_eq!(
                samples.len(),
                width * height * if layout.is_grayscale() { 1 } else { 3 }
            );
            if layout.has_alpha() {
                let alpha = p.with_file_name(format!(
                    "{}-ec1.pfm",
                    p.file_stem().unwrap().to_str().unwrap()
                ));
                samples.extend(pfm::read_pfm_samples(&alpha, (width, height)));
            }
            assert!(samples.iter().all(|v| v.is_finite()));
            samples
        })
        .collect();
    [primary, native]
}
#[test]
fn progressive_animation_entries_preserve_every_displayed_frame() {
    use jxl_encoder::{AnimationFrame, AnimationParams, EncoderMode};
    let root = directory();
    let fixture = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    for (width, height) in [(64, 64), (259, 133)] {
        let frames: Vec<Vec<u8>> = (0..2)
            .map(|frame| {
                (0..height)
                    .flat_map(|y| {
                        let fixture = &fixture;
                        (0..width).flat_map(move |x| {
                            fixture
                                .get_pixel(
                                    (x as u32 + frame * 3) % fixture.width(),
                                    (y as u32 + frame) % fixture.height(),
                                )
                                .0
                        })
                    })
                    .collect()
            })
            .collect();
        for layout in [
            PixelLayout::Rgb8,
            PixelLayout::Rgba16,
            PixelLayout::GrayAlpha8,
        ] {
            let pixels: Vec<_> = frames.iter().map(|p| typed_pixels(p, layout)).collect();
            for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
                for entry in ["legacy", "request"] {
                    for associated in [false, true] {
                        if associated && !layout.has_alpha() {
                            continue;
                        }
                        let out = root.join(format!(
                            "{width}x{height}-{layout:?}-{mode:?}-{entry}-assoc{associated}"
                        ));
                        std::fs::create_dir(&out).unwrap();
                        let supplied: Vec<_> = pixels
                            .iter()
                            .map(|p| {
                                if associated {
                                    associated_pixels(p, layout)
                                } else {
                                    p.clone()
                                }
                            })
                            .collect();
                        let frames = [
                            AnimationFrame::new(&supplied[0], 10),
                            AnimationFrame::new(&supplied[1], 20),
                        ];
                        let params = AnimationParams {
                            premultiplied_alpha: associated,
                            ..Default::default()
                        };
                        let config = LossyConfig::new(1.0)
                            .with_effort(4)
                            .with_threads(1)
                            .with_mode(mode)
                            .with_strategy(EncoderStrategy::Libjxl)
                            .with_force_strategy(Some(3))
                            .with_auto_delta_frames(false);
                        let encode = |progressive| {
                            let config = config.clone().with_progressive(progressive);
                            if entry == "legacy" {
                                config.encode_animation(
                                    width as u32,
                                    height as u32,
                                    layout,
                                    &params,
                                    &frames,
                                )
                            } else {
                                config
                                    .encode_request(width as u32, height as u32, layout)
                                    .encode_animation(&params, &frames)
                            }
                            .unwrap()
                        };
                        let baseline = render_animation(
                            &out,
                            "single",
                            &encode(ProgressiveMode::Single),
                            width,
                            height,
                            layout,
                        );
                        for progressive in [
                            ProgressiveMode::QuantizedAcFullAc,
                            ProgressiveMode::DcVlfLfAc,
                        ] {
                            let decoded = render_animation(
                                &out,
                                &format!("{progressive:?}"),
                                &encode(progressive),
                                width,
                                height,
                                layout,
                            );
                            for (actual, expected) in decoded.iter().zip(&baseline) {
                                assert_eq!(actual.len(), expected.len());
                                for (frame, (a, b)) in actual.iter().zip(expected).enumerate() {
                                    assert_eq!(a.len(), b.len());
                                    for (sample, (a, b)) in a.iter().zip(b).enumerate() {
                                        assert_eq!(
                                            a.to_bits(),
                                            b.to_bits(),
                                            "{} frame{frame} sample{sample}",
                                            out.display()
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(all(feature = "__pre_quantized", feature = "__internals"))]
#[test]
fn progressive_prepared_engines_preserve_final_coefficients() {
    use jxl_encoder::__pre_quantized::{DistanceParams, EncoderPrecomputed, VarDctEncoder};
    use jxl_encoder::EncoderMode;
    use jxl_encoder::api::ExtraChannel;
    let root = directory();
    let fixture = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    for (width, height) in [(64, 64), (259, 133)] {
        let rgb: Vec<u8> = (0..height)
            .flat_map(|y| {
                let fixture = &fixture;
                (0..width).flat_map(move |x| {
                    fixture
                        .get_pixel(x as u32 % fixture.width(), y as u32 % fixture.height())
                        .0
                })
            })
            .collect();
        let linear: Vec<f32> = rgb.iter().map(|v| f32::from(*v) / 255.0).collect();
        let alpha: Vec<u8> = rgb.as_chunks::<3>().0.iter().map(|p| p[1]).collect();
        for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
            for entry in ["precomputed", "prequantized"] {
                for with_alpha in [false, true] {
                    let out = root.join(format!(
                        "{width}x{height}-{mode:?}-{entry}-alpha{with_alpha}"
                    ));
                    std::fs::create_dir(&out).unwrap();
                    std::fs::write(out.join("input.rgb8"), &rgb).unwrap();
                    let mut encoder = VarDctEncoder::new(1.0);
                    // These engine APIs expose no named-strategy setter.
                    encoder.profile = LossyConfig::new(1.0).with_effort(4).resolved_profile();
                    encoder.encoder_mode = mode;
                    encoder.effort = 4;
                    encoder.force_strategy = Some(3);
                    encoder.enable_patches = false;
                    let pre = EncoderPrecomputed::compute(
                        width,
                        height,
                        &linear,
                        encoder.distance,
                        encoder.cfl_enabled,
                        encoder.ac_strategy_enabled,
                        encoder.pixel_domain_loss,
                        encoder.enable_noise,
                        encoder.enable_denoise,
                        encoder.enable_gaborish,
                        encoder.force_strategy,
                        &encoder.profile,
                        encoder.color_encoding.as_ref(),
                    )
                    .unwrap();
                    let mut qf = vec![1; pre.xsize_blocks * pre.ysize_blocks];
                    let params =
                        DistanceParams::compute_for_profile(encoder.distance, &encoder.profile);
                    let transform = encoder
                        .transform_and_quantize_for_test(&pre, &mut qf, &params)
                        .unwrap();
                    let extras = [ExtraChannel::from_alpha_buf(&alpha, false)];
                    let layout = if with_alpha {
                        PixelLayout::Rgba8
                    } else {
                        PixelLayout::Rgb8
                    };
                    let mut baseline: Option<[Vec<u32>; 2]> = None;
                    for progressive in [
                        ProgressiveMode::Single,
                        ProgressiveMode::QuantizedAcFullAc,
                        ProgressiveMode::DcVlfLfAc,
                    ] {
                        encoder.progressive = progressive;
                        let bytes = match (entry, with_alpha) {
                            ("precomputed", false) => encoder.encode_from_precomputed(&pre, &qf),
                            ("precomputed", true) => {
                                encoder.encode_from_precomputed_with_extras(&pre, &qf, &extras)
                            }
                            ("prequantized", false) => encoder.encode_from_pre_quantized_ac(
                                &pre,
                                &qf,
                                &transform.quant_dc,
                                &transform.quant_ac,
                                &transform.nzeros,
                                &transform.raw_nzeros,
                            ),
                            ("prequantized", true) => encoder
                                .encode_from_pre_quantized_ac_with_extras(
                                    &pre,
                                    &qf,
                                    &transform.quant_dc,
                                    &transform.quant_ac,
                                    &transform.nzeros,
                                    &transform.raw_nzeros,
                                    &extras,
                                ),
                            _ => unreachable!(),
                        }
                        .unwrap();
                        let actual = render(
                            &out,
                            &bytes,
                            &format!("{progressive:?}"),
                            width,
                            height,
                            layout,
                        );
                        if let Some(expected) = &baseline {
                            for (a, b) in actual.iter().zip(expected) {
                                assert_eq!(
                                    a.iter().map(|v| v.to_bits()).collect::<Vec<_>>().as_slice(),
                                    b.as_slice(),
                                    "{} {progressive:?}",
                                    out.display()
                                );
                            }
                        } else {
                            baseline = Some(
                                actual.map(|v| v.into_iter().map(f32::to_bits).collect::<Vec<_>>()),
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn progressive_producer_options_preserve_final_pixels() {
    use jxl_encoder::{Buffering, EncoderMode};
    let root = directory();
    let fixture = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    let (width, height) = (259, 133);
    let pixels: Vec<u8> = (0..height)
        .flat_map(|y| {
            let fixture = &fixture;
            (0..width).flat_map(move |x| {
                fixture
                    .get_pixel(x as u32 % fixture.width(), y as u32 % fixture.height())
                    .0
            })
        })
        .collect();
    for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
        for strategy in [EncoderStrategy::Libjxl, EncoderStrategy::Zenjxl] {
            let base = LossyConfig::new(1.0)
                .with_effort(4)
                .with_threads(1)
                .with_mode(mode)
                .with_strategy(strategy.clone())
                .with_force_strategy(Some(3));
            let mut cases = vec![
                ("e1", base.clone().with_effort(1)),
                ("e2", base.clone().with_effort(2)),
                ("e7", base.clone().with_effort(7)),
                ("e8-loop", base.clone().with_effort(8)),
                ("lf-frame", base.clone().with_lf_frame(true)),
                ("resampling2", base.clone().with_resampling(2)),
                ("threads2", base.clone().with_threads(2)),
                (
                    "buffer-full",
                    base.clone().with_buffering(Buffering::FullBuffered),
                ),
                (
                    "buffer-threshold",
                    base.clone().with_buffering(Buffering::Threshold2048),
                ),
                (
                    "buffer-output",
                    base.clone().with_buffering(Buffering::BufferedOutput),
                ),
            ];
            #[cfg(feature = "butteraugli-loop")]
            cases.push(("explicit-loop", base.clone().with_butteraugli_iters(2)));
            for (name, config) in cases.drain(..) {
                check_config(
                    &root.join(format!("{strategy:?}-{mode:?}-{name}")),
                    config,
                    PixelCase {
                        pixels: &pixels,
                        width,
                        height,
                        layout: PixelLayout::Rgb8,
                        entry: "request",
                    },
                );
            }
            let rgba = typed_pixels(&pixels, PixelLayout::Rgba8);
            check_config(
                &root.join(format!("{strategy:?}-{mode:?}-canonical")),
                base.with_canonicalize_input(true),
                PixelCase {
                    pixels: &rgba,
                    width,
                    height,
                    layout: PixelLayout::Rgba8,
                    entry: "request",
                },
            );
        }
    }
}

#[test]
fn progressive_multi_dc_groups_and_already_downsampled_preserve_final_pixels() {
    use jxl_encoder::EncoderMode;
    let root = directory();
    let fixture = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    for (width, height, factor) in [(18433, 9, 1), (259, 133, 2)] {
        let pixels: Vec<u8> = (0..height)
            .flat_map(|y| {
                let fixture = &fixture;
                (0..width).flat_map(move |x| {
                    fixture
                        .get_pixel(x as u32 % fixture.width(), y as u32 % fixture.height())
                        .0
                })
            })
            .collect();
        for strategy in [EncoderStrategy::Libjxl, EncoderStrategy::Zenjxl] {
            for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
                let out = root.join(format!(
                    "{strategy:?}-{mode:?}-{width}x{height}-factor{factor}"
                ));
                std::fs::create_dir(&out).unwrap();
                std::fs::write(out.join("input.rgb8"), &pixels).unwrap();
                let mut baseline: Option<[Vec<u32>; 2]> = None;
                for progressive in [
                    ProgressiveMode::Single,
                    ProgressiveMode::QuantizedAcFullAc,
                    ProgressiveMode::DcVlfLfAc,
                ] {
                    let config = LossyConfig::new(1.0)
                        .with_effort(4)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_force_strategy(Some(3))
                        .with_progressive(progressive)
                        .with_resampling(factor)
                        .with_already_downsampled(factor > 1);
                    let bytes = config
                        .encode(&pixels, width as u32, height as u32, PixelLayout::Rgb8)
                        .unwrap();
                    let image = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(&bytes))
                        .unwrap();
                    if factor == 1 {
                        assert_eq!(image.frame(0).unwrap().header().num_lf_groups(), 10);
                    }
                    let actual = render(
                        &out,
                        &bytes,
                        &format!("{progressive:?}"),
                        width * factor as usize,
                        height * factor as usize,
                        PixelLayout::Rgb8,
                    );
                    if let Some(expected) = &baseline {
                        for (a, b) in actual.iter().zip(expected) {
                            assert_eq!(
                                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                *b,
                                "{} {progressive:?}",
                                out.display()
                            );
                        }
                    } else {
                        baseline = Some(
                            actual.map(|v| v.into_iter().map(f32::to_bits).collect::<Vec<_>>()),
                        );
                    }
                }
            }
        }
    }
}

#[cfg(feature = "corpus-tests")]
#[test]
fn progressive_corpus_patches_preserve_final_pixels() {
    use jxl_encoder::EncoderMode;
    let root = directory();
    let input = jxl_encoder::test_helpers::corpus_dir().join("gb82-sc/imac_g3.png");
    let source = std::fs::read(&input).expect("caller-selected imac_g3 PNG is required");
    std::fs::write(root.join("source.png"), &source).unwrap();
    let image = image::load_from_memory(&source).unwrap().to_rgb8();
    let (width, height) = (image.width() as usize, image.height() as usize);
    for strategy in [EncoderStrategy::Libjxl, EncoderStrategy::Zenjxl] {
        for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
            let out = root.join(format!("{strategy:?}-{mode:?}"));
            std::fs::create_dir(&out).unwrap();
            let mut baseline: Option<[Vec<u32>; 2]> = None;
            for progressive in [
                ProgressiveMode::Single,
                ProgressiveMode::QuantizedAcFullAc,
                ProgressiveMode::DcVlfLfAc,
            ] {
                let config = LossyConfig::new(1.0)
                    .with_effort(7)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_patches(true)
                    .with_force_strategy(Some(3))
                    .with_progressive(progressive);
                let bytes = config
                    .encode(
                        image.as_raw(),
                        width as u32,
                        height as u32,
                        PixelLayout::Rgb8,
                    )
                    .unwrap();
                let decoded = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&bytes))
                    .unwrap();
                assert_eq!(
                    decoded.frame(0).unwrap().header().frame_type as u8,
                    2,
                    "actual winning patch reference required"
                );
                let actual = render(
                    &out,
                    &bytes,
                    &format!("{progressive:?}"),
                    width,
                    height,
                    PixelLayout::Rgb8,
                );
                if let Some(expected) = &baseline {
                    for (a, b) in actual.iter().zip(expected) {
                        assert_eq!(
                            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>().as_slice(),
                            b.as_slice(),
                            "{} {progressive:?}",
                            out.display()
                        );
                    }
                } else {
                    baseline =
                        Some(actual.map(|v| v.into_iter().map(f32::to_bits).collect::<Vec<_>>()));
                }
            }
        }
    }
}

#[test]
fn progressive_reference_only_and_cropped_animation_frames() {
    use jxl_encoder::{AnimationFrame, AnimationParams};
    let root = directory();
    let fixture = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    let mut failures = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(64, 64), (259, 133)] {
            let rgb: Vec<u8> = (0..height)
                .flat_map(|y| {
                    let fixture = &fixture;
                    (0..width).flat_map(move |x| {
                        fixture
                            .get_pixel(x as u32 % fixture.width(), y as u32 % fixture.height())
                            .0
                    })
                })
                .collect();
            let mut second = rgb.clone();
            for y in 16..32 {
                for x in 16..32 {
                    second[(y * width + x) * 3] ^= 0x7f;
                }
            }
            for layout in [
                PixelLayout::Rgb8,
                PixelLayout::Rgba16,
                PixelLayout::GrayAlpha8,
            ] {
                for associated in [false, true] {
                    if associated && !layout.has_alpha() {
                        continue;
                    }
                    let first = typed_pixels(&rgb, layout);
                    let second = typed_pixels(&second, layout);
                    let first = if associated {
                        associated_pixels(&first, layout)
                    } else {
                        first
                    };
                    let second = if associated {
                        associated_pixels(&second, layout)
                    } else {
                        second
                    };
                    let frames = [
                        AnimationFrame::new(&second, 0)
                            .with_reference_only(true)
                            .with_save_as_reference(2),
                        AnimationFrame::new(&first, 10),
                        AnimationFrame::new(&first, 0)
                            .with_reference_only(true)
                            .with_save_as_reference(2),
                        AnimationFrame::new(&second, 20),
                    ];
                    for entry in ["legacy", "request"] {
                        let out = root.join(format!(
                        "{policy}-{mode:?}-{width}x{height}-{layout:?}-{entry}-assoc{associated}"
                    ));
                        std::fs::create_dir(&out).unwrap();
                        std::fs::write(out.join("input0.bin"), &first).unwrap();
                        std::fs::write(out.join("input1.bin"), &second).unwrap();
                        let mut baseline: Option<[Vec<Vec<f32>>; 2]> = None;
                        for progressive in [
                            ProgressiveMode::Single,
                            ProgressiveMode::QuantizedAcFullAc,
                            ProgressiveMode::DcVlfLfAc,
                        ] {
                            let config = LossyConfig::new(1.0)
                                .with_effort(4)
                                .with_threads(1)
                                .with_strategy(strategy.clone())
                                .with_mode(mode)
                                .with_force_strategy(Some(3))
                                .with_progressive(progressive)
                                .with_auto_delta_frames(false)
                                .with_patches(false);
                            let params = AnimationParams {
                                premultiplied_alpha: associated,
                                ..Default::default()
                            };
                            let bytes = if entry == "legacy" {
                                config.encode_animation(
                                    width as u32,
                                    height as u32,
                                    layout,
                                    &params,
                                    &frames,
                                )
                            } else {
                                config
                                    .encode_request(width as u32, height as u32, layout)
                                    .encode_animation(&params, &frames)
                            }
                            .unwrap();
                            let actual = std::panic::catch_unwind(|| {
                                render_animation(
                                    &out,
                                    &format!("{progressive:?}"),
                                    &bytes,
                                    width,
                                    height,
                                    layout,
                                )
                            });
                            match actual {
                                Ok(actual) => {
                                    let image = jxl_oxide::JxlImage::builder()
                                        .read(std::io::Cursor::new(&bytes))
                                        .unwrap();
                                    for index in [0, 2] {
                                        let header = image.frame(index).unwrap().header();
                                        assert_eq!(header.frame_type as u8, 2);
                                        assert_eq!(header.passes.num_passes, 1);
                                    }
                                    let wanted = match progressive {
                                        ProgressiveMode::Single => 1,
                                        ProgressiveMode::QuantizedAcFullAc => 2,
                                        ProgressiveMode::DcVlfLfAc => 3,
                                    };
                                    for index in [1, 3] {
                                        assert_eq!(
                                            image.frame(index).unwrap().header().passes.num_passes,
                                            wanted,
                                            "regular frame must restore requested schedule"
                                        );
                                    }
                                    let crop = image.frame(3).unwrap().header();
                                    assert!(
                                        crop.width < width as u32 || crop.height < height as u32,
                                        "actual crop required"
                                    );
                                    if let Some(expected) = &baseline {
                                        for (a, b) in actual.iter().zip(expected) {
                                            for (a, b) in a.iter().zip(b) {
                                                assert_eq!(
                                                    a.iter()
                                                        .map(|v| v.to_bits())
                                                        .collect::<Vec<_>>(),
                                                    b.iter()
                                                        .map(|v| v.to_bits())
                                                        .collect::<Vec<_>>(),
                                                    "{} {progressive:?}",
                                                    out.display()
                                                );
                                            }
                                        }
                                    } else {
                                        baseline = Some(actual);
                                    }
                                }
                                Err(_) => failures.push(format!(
                                    "{} {progressive:?} failed rendering",
                                    out.display()
                                )),
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[cfg(all(feature = "rate-control", feature = "__pre_quantized"))]
#[test]
fn progressive_rate_control_and_region_precompute_preserve_final_pixels() {
    use jxl_encoder::__pre_quantized::VarDctEncoder;
    use jxl_encoder::vardct::RateControlConfig;
    use jxl_encoder::{Buffering, EncoderMode};
    let root = directory();
    let fixture = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    for (width, height) in [(64, 64), (18433, 9)] {
        let pixels: Vec<u8> = (0..height)
            .flat_map(|y| {
                let fixture = &fixture;
                (0..width).flat_map(move |x| {
                    fixture
                        .get_pixel(x as u32 % fixture.width(), y as u32 % fixture.height())
                        .0
                })
            })
            .collect();
        let linear: Vec<f32> = pixels.iter().map(|v| f32::from(*v) / 255.0).collect();
        for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
            for buffering in [
                Buffering::FullBuffered,
                Buffering::Threshold2048,
                Buffering::BufferedOutput,
                Buffering::FullStreaming,
            ] {
                let out = root.join(format!("{width}x{height}-{mode:?}-{buffering:?}"));
                std::fs::create_dir(&out).unwrap();
                std::fs::write(out.join("input.rgb8"), &pixels).unwrap();
                let mut encoder = VarDctEncoder::new(1.0);
                // The direct engine exposes its default policy, not a named-strategy setter.
                encoder.profile = LossyConfig::new(1.0).with_effort(4).resolved_profile();
                encoder.effort = 4;
                encoder.encoder_mode = mode;
                encoder.force_strategy = Some(3);
                encoder.enable_patches = false;
                encoder.buffering = buffering;
                let control = RateControlConfig {
                    max_iterations: 1,
                    ..Default::default()
                };
                let mut baseline: Option<([Vec<u32>; 2], usize)> = None;
                for (progressive, wanted) in [
                    (ProgressiveMode::Single, 1),
                    (ProgressiveMode::QuantizedAcFullAc, 2),
                    (ProgressiveMode::DcVlfLfAc, 3),
                ] {
                    encoder.progressive = progressive;
                    let (bytes, iterations) = encoder
                        .encode_with_rate_control_config(width, height, &linear, &control)
                        .unwrap();
                    std::fs::write(
                        out.join(format!("{progressive:?}.iterations")),
                        format!("{iterations}\n"),
                    )
                    .unwrap();
                    let image = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(&bytes))
                        .unwrap();
                    assert_eq!(image.frame(0).unwrap().header().passes.num_passes, wanted);
                    let actual = render(
                        &out,
                        &bytes,
                        &format!("{progressive:?}"),
                        width,
                        height,
                        PixelLayout::Rgb8,
                    )
                    .map(|v| v.into_iter().map(f32::to_bits).collect::<Vec<_>>());
                    if let Some((expected, expected_iterations)) = &baseline {
                        assert_eq!(&actual, expected, "{} {progressive:?}", out.display());
                        assert_eq!(iterations, *expected_iterations);
                    } else {
                        baseline = Some((actual, iterations));
                    }
                }
            }
        }
    }
}

#[test]
fn progressive_typed_animation_layout_inventory() {
    use jxl_encoder::{AnimationFrame, AnimationParams, EncoderMode};
    let root = directory();
    let fixture = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    for (index, layout) in non_cmyk_lossy_layouts().into_iter().enumerate() {
        let (width, height) = if index % 2 == 0 { (64, 64) } else { (259, 133) };
        let rgb: Vec<u8> = (0..height)
            .flat_map(|y| {
                let fixture = &fixture;
                (0..width).flat_map(move |x| {
                    fixture
                        .get_pixel(x as u32 % fixture.width(), y as u32 % fixture.height())
                        .0
                })
            })
            .collect();
        let shifted: Vec<u8> = rgb.iter().map(|v| v.wrapping_add(17)).collect();
        let first = typed_pixels(&rgb, layout);
        let second = typed_pixels(&shifted, layout);
        let frames = [
            AnimationFrame::new(&second, 0)
                .with_reference_only(true)
                .with_save_as_reference(2),
            AnimationFrame::new(&first, 10),
            AnimationFrame::new(&first, 0)
                .with_reference_only(true)
                .with_save_as_reference(2),
            AnimationFrame::new(&second, 20),
        ];
        for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
            for entry in ["legacy", "request"] {
                let out = root.join(format!("{layout:?}-{mode:?}-{entry}-{width}x{height}"));
                std::fs::create_dir(&out).unwrap();
                std::fs::write(out.join("input0.bin"), &first).unwrap();
                std::fs::write(out.join("input1.bin"), &second).unwrap();
                let mut baseline: Option<[Vec<Vec<u32>>; 2]> = None;
                for progressive in [
                    ProgressiveMode::Single,
                    ProgressiveMode::QuantizedAcFullAc,
                    ProgressiveMode::DcVlfLfAc,
                ] {
                    let config = LossyConfig::new(1.0)
                        .with_effort(4)
                        .with_threads(1)
                        .with_strategy(EncoderStrategy::Libjxl)
                        .with_mode(mode)
                        .with_force_strategy(Some(3))
                        .with_progressive(progressive)
                        .with_auto_delta_frames(false)
                        .with_patches(false);
                    let params = AnimationParams::default();
                    let bytes = if entry == "legacy" {
                        config.encode_animation(
                            width as u32,
                            height as u32,
                            layout,
                            &params,
                            &frames,
                        )
                    } else {
                        config
                            .encode_request(width as u32, height as u32, layout)
                            .encode_animation(&params, &frames)
                    }
                    .unwrap();
                    let actual = render_animation(
                        &out,
                        &format!("{progressive:?}"),
                        &bytes,
                        width,
                        height,
                        layout,
                    )
                    .map(|frames| {
                        frames
                            .into_iter()
                            .map(|p| p.into_iter().map(f32::to_bits).collect::<Vec<_>>())
                            .collect::<Vec<_>>()
                    });
                    if let Some(expected) = &baseline {
                        assert_eq!(&actual, expected, "{} {progressive:?}", out.display());
                    } else {
                        baseline = Some(actual);
                    }
                }
            }
        }
    }
}
